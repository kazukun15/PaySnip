import streamlit as st
import pandas as pd
import io, zipfile, re, time
import numpy as np
from datetime import datetime
from typing import List, Dict
from pypdf import PdfReader, PdfWriter
from PIL import Image
import pytesseract

# ── アプリ設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("📄 支払通知書抽出ツール")

# ── サイドバー ────────────────────────────────────
st.sidebar.header("ファイルアップロード")
pdf_file = st.sidebar.file_uploader("PDFファイル (.pdf)", type="pdf")
csv_file = st.sidebar.file_uploader("CSVファイル (.csv)", type="csv")
st.sidebar.markdown("---")
st.sidebar.header("オプション設定")
enable_refine = st.sidebar.checkbox("Gemini補正を有効にする", value=False)
action_preview = st.sidebar.button("プレビュー")
action_extract = st.sidebar.button("抽出")

# ── データロード関数 ─────────────────────────────────
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    for enc in ("utf-8", "cp932", "shift-jis"):
        try:
            file.seek(0)
            return pd.read_csv(file, dtype=str, encoding=enc)
        except Exception:
            continue
    st.error("CSV読み込み失敗: エンコーディングを確認してください。")
    st.stop()

@st.cache_data
def load_pdf_bytes(file) -> bytes:
    file.seek(0)
    return file.read()

# ── リソース初期化（遅延ロード） ─────────────────────────────
@st.cache_resource
def init_pdf_renderer():
    import fitz  # PyMuPDF
    return fitz

@st.cache_resource
def init_gemini_model(api_key: str):
    if not api_key:
        return None
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    except Exception:
        return None

# ── OCR 補助関数（Tesseract） ─────────────────────────────
def ocr_page(fitz, page_bytes: bytes, page_index: int) -> str:
    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        return pytesseract.image_to_string(img, lang='jpn')
    except:
        return pytesseract.image_to_string(img)

# ── テキスト正規化・補正 ─────────────────────────────
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)

def refine_text(raw: str, page_num: int, model) -> str:
    if not model:
        return raw
    try:
        prompt = f"PDFの{page_num}ページから抽出された支払通知書テキストを自然な日本語に修正してください:\n{raw}"
        res = model.generate_content(prompt)
        return getattr(res, 'text', raw)
    except:
        return raw

# ── マッチング処理 ─────────────────────────────────
def find_matches(reader: PdfReader, pdf_bytes: bytes, names: List[str], accounts: List[str], fitz, model) -> List[Dict]:
    results = []
    for idx, page in enumerate(reader.pages, start=1):
        # テキストレイヤー抽出
        raw_text = page.extract_text() or ""
        # 常にOCRテキストも結合してマッチング精度を向上
        ocr_text = ocr_page(fitz, pdf_bytes, idx-1)
        combined = raw_text + "\n" + ocr_text
        # 補正オプション
        refined = refine_text(combined, idx, model)
        norm = normalize_text(refined)
        found = None
        # 名前マッチ
        for name in names:
            if normalize_text(name) in norm:
                found = name
                break
        # 口座番号マッチ
        if not found:
            digits = re.sub(r"\D", "", refined)
            for acc in accounts:
                if re.sub(r"\D", "", acc) in digits:
                    found = acc
                    break
        if found:
            results.append({"page": idx, "match": found})
    return results

# ── メイン処理 ─────────────────────────────────────
if not pdf_file or not csv_file:
    st.warning("PDFとCSVをアップロードしてください。")
    st.stop()

csv_df = load_csv(csv_file)
pdf_bytes = load_pdf_bytes(pdf_file)
pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

names = csv_df.get("相手方", pd.Series()).dropna().str.strip().tolist()
accounts = sum([csv_df.get(c, pd.Series()).dropna().str.strip().tolist() for c in ["口座番号１","口座番号２","口座番号３"]], [])

st.subheader("CSV プレビュー")
st.dataframe(csv_df.head(5))
st.write(f"PDFページ数: {len(pdf_reader.pages)}")

if action_preview:
    fitz = init_pdf_renderer()
    model = init_gemini_model(st.secrets.get("gemini", {}).get("api_key", "")) if enable_refine else None
    with st.spinner("プレビュー中…"):
        t0 = time.time()
        preview = find_matches(pdf_reader, pdf_bytes, names, accounts, fitz, model)
        elapsed = time.time() - t0
    st.success(f"プレビュー完了 ({elapsed:.2f}s)")
    st.table(preview or [])

if action_extract:
    fitz = init_pdf_renderer()
    model = init_gemini_model(st.secrets.get("gemini", {}).get("api_key", "")) if enable_refine else None
    with st.spinner("抽出中…"):
        t0 = time.time()
        matches = find_matches(pdf_reader, pdf_bytes, names, accounts, fitz, model)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for m in matches:
                pg = m['page'] - 1
                writer = PdfWriter()
                writer.add_page(pdf_reader.pages[pg])
                b = io.BytesIO(); writer.write(b)
                safe = re.sub(r"[\\/:*?\"<>|]", "_", m['match'])
                fname = f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{m['page']}.pdf"
                zf.writestr(fname, b.getvalue())
        buf.seek(0)
        elapsed = time.time() - t0
    if matches:
        st.success(f"抽出完了 ({elapsed:.2f}s) - {len(matches)} ページを出力しました。")
        st.download_button("ZIPダウンロード", buf, file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip")
        st.dataframe(matches)
    else:
        st.warning(f"一致なし ({elapsed:.2f}s)")
