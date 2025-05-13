import streamlit as st
import pandas as pd
import io
import zipfile
import re
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from pypdf import PdfReader, PdfWriter
import fitz  # PyMuPDF for PDF rendering
from PIL import Image
import easyocr

# ── アプリ設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("📄 支払通知書抽出ツール")

# ── サイドバー ────────────────────────────────────
st.sidebar.header("ファイルアップロード")
pdf_file = st.sidebar.file_uploader("PDFファイル (.pdf)", type="pdf")
csv_file = st.sidebar.file_uploader("CSVファイル (.csv)", type="csv")
st.sidebar.markdown("---")
# Gemini補正オプション
st.sidebar.header("オプション設定")
enable_refine = st.sidebar.checkbox("Geminiによるテキスト補正を有効にする", value=False)
action_preview = st.sidebar.button("プレビュー")
action_extract = st.sidebar.button("抽出")

# ── Geminiモデル初期化 ──────────────────────────────
@st.cache_resource
def init_gemini_model(api_key: str) -> Optional:
    import google.generativeai as genai
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    except Exception:
        return None

gemini_api_key = st.secrets.get("gemini", {}).get("api_key", "")
model = init_gemini_model(gemini_api_key)

# ── OCRリーダー初期化 (pure Python via easyocr) ─────────────────────────
@st.cache_resource
def get_easyocr_reader():
    # GPU=FalseでCPUモード
    return easyocr.Reader(['ja'], gpu=False)
ocr_reader = get_easyocr_reader()

# ── ユーティリティ関数 ─────────────────────────────
def normalize_text(text: str) -> str:
    """空白と改行を除去し比較用に正規化"""
    return re.sub(r"\s+", "", text)

def refine_text(raw: str, page: int) -> str:
    """Gemini APIでテキストを補正。失敗時は生テキストを返す"""
    if not model or not enable_refine:
        return raw
    try:
        prompt = (
            f"PDFの{page}ページから抽出された支払通知書テキストを、誤字脱字なく自然な日本語に修正してください。\n" + raw
        )
        res = model.generate_content(prompt)
        return res.text
    except Exception:
        return raw

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """UTF-8, CP932, Shift-JISを順に試行してCSVを読み込む"""
    for enc in ("utf-8", "cp932", "shift-jis"):
        try:
            file.seek(0)
            return pd.read_csv(file, dtype=str, encoding=enc)
        except Exception:
            continue
    st.error("CSVの読み込みに失敗しました。Encodingを確認してください。")
    st.stop()

@st.cache_data
def load_pdf_bytes(file) -> bytes:
    """アップロードPDFからバイト列を読み込む"""
    file.seek(0)
    return file.read()

# ── OCR用関数: PyMuPDF + easyocr ─────────────────────────
def ocr_page(fitz_doc: fitz.Document, page_num: int) -> str:
    """ページを画像化してOCRテキストを取得"""
    page = fitz_doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    arr = np.array(img)
    texts = ocr_reader.readtext(arr, detail=0)
    return "\n".join(texts)

# ── マッチング関数 ─────────────────────────────────
def find_matches(
    reader: PdfReader,
    fitz_doc: fitz.Document,
    names: List[str],
    accounts: List[str]
) -> List[Dict]:
    """テキスト層＋OCR併用でマッチング"""
    results = []
    for idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        # テキストが短い場合はOCRを併用
        if len(raw.strip()) < 20:
            raw += "\n" + ocr_page(fitz_doc, idx-1)
        text = refine_text(raw, idx)
        norm = normalize_text(text)
        found = None
        # 名前マッチ
        for name in names:
            if normalize_text(name) in norm:
                found = name
                break
        # 補助: 口座番号
        if not found:
            digits = re.sub(r"\D", "", text)
            for acc in accounts:
                if re.sub(r"\D", "", acc) in digits:
                    found = acc
                    break
        if found:
            results.append({"page": idx, "match": found})
    return results

# ── アプリ本体 ─────────────────────────────────────
if not pdf_file or not csv_file:
    st.warning("PDFとCSVをアップロードしてください。")
    st.stop()

# データ読込
csv_df = load_csv(csv_file)
pdf_bytes = load_pdf_bytes(pdf_file)
pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

names = csv_df.get("相手方", pd.Series()).dropna().str.strip().tolist()
accounts = sum(
    [csv_df.get(col, pd.Series()).dropna().str.strip().tolist() for col in ["口座番号１","口座番号２","口座番号３"]], []
)

# プレビュー表示
st.subheader("CSVサマリ")
st.dataframe(csv_df.head(5))
st.write(f"PDFページ数: {len(pdf_reader.pages)}")

if action_preview:
    with st.spinner("プレビュー中…"):
        t0 = time.time()
        preview = find_matches(pdf_reader, fitz_doc, names, accounts)
        elapsed = time.time() - t0
    st.success(f"プレビュー完了 ({elapsed:.2f}s)")
    if preview:
        st.table(pd.DataFrame(preview))
    else:
        st.info("一致するページがありません。")

# 抽出処理
if action_extract:
    with st.spinner("抽出中…"):
        t0 = time.time()
        matches = find_matches(pdf_reader, fitz_doc, names, accounts)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for item in matches:
                pg = item['page'] - 1
                writer = PdfWriter()
                writer.add_page(pdf_reader.pages[pg])
                b = io.BytesIO(); writer.write(b)
                safe = re.sub(r"[\\/:*?\"<>|]", "_", item['match'])
                fname = f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{item['page']}.pdf"
                zf.writestr(fname, b.getvalue())
        buf.seek(0)
        elapsed = time.time() - t0
    if matches:
        st.success(f"抽出完了 ({elapsed:.2f}s) - {len(matches)}ページを出力しました。")
        st.download_button("ZIPダウンロード", buf, file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip")
        st.subheader("抽出結果一覧")
        st.dataframe(pd.DataFrame(matches))
    else:
        st.warning(f"抽出対象が見つかりませんでした。 ({elapsed:.2f}s) ")
