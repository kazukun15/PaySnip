import streamlit as st
import pandas as pd
import io
import zipfile
import re
import time
from datetime import datetime
from pypdf import PdfReader, PdfWriter
import pytesseract
from pdf2image import convert_from_bytes, exceptions as pdf2image_exceptions
import google.generativeai as genai
from google.api_core.exceptions import InternalServerError

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="OCR＋Gemini 支払通知書抽出", layout="wide")

# ── サイドバー：設定・アップロード ─────────────────────────
st.sidebar.header("設定")
# secrets.toml に [gemini] api_key="..." を設定済み
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.secrets.get("gemini", {}).get("api_key", ""))
model = None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

st.sidebar.header("ファイルアップロード")
pdf_file = st.sidebar.file_uploader("PDFファイルを選択", type="pdf")
csv_file = st.sidebar.file_uploader("CSVファイルを選択", type="csv")

# 操作ボタン
preview_btn = st.sidebar.button("プレビュー実行")
extract_btn = st.sidebar.button("抽出実行")

# ── 共通関数 ─────────────────────────────────────
def normalize_text(text: str) -> str:
    return "".join(text.split())

def refine_with_gemini(raw: str) -> str:
    if not model: return raw
    try:
        prompt = f"以下のOCRテキストを日本語として正確に修正してください:\n```{raw}```"
        response = model.generate_content(prompt)
        return response.text
    except InternalServerError:
        return raw
    except Exception:
        return raw

@st.cache_data
def extract_text(pdf_bytes: bytes, page_num: int) -> str:
    try:
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=300)
        return pytesseract.image_to_string(images[0], lang="jpn")
    except pdf2image_exceptions.PDFInfoNotInstalledError:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return reader.pages[page_num-1].extract_text() or ""
    except Exception:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return reader.pages[page_num-1].extract_text() or ""

# ── メイン処理 ───────────────────────────────────
st.title("OCR＋Gemini 支払通知書抽出ツール")

if not pdf_file or not csv_file:
    st.info("左側のサイドバーでPDFとCSVをアップロードしてください。")
    st.stop()

# CSV読み込み
for enc in ("utf-8","cp932"):
    try:
        csv_file.seek(0)
        df = pd.read_csv(csv_file, dtype=str, encoding=enc)
        break
    except Exception:
        continue
else:
    st.error("CSVの読み込みに失敗しました。")
    st.stop()

st.subheader("CSV プレビュー")
st.dataframe(df)

# ターゲットリスト
name_targets = df.get("相手方", pd.Series()).dropna().astype(str).str.strip().tolist()
account_targets = []
for col in ("口座番号１","口座番号２","口座番号３"):
    if col in df.columns:
        account_targets += df[col].dropna().astype(str).str.strip().tolist()

# PDF bytes
pdf_bytes = pdf_file.read()

# プレビュー処理
def run_preview():
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    progress = st.progress(0)
    matches = []
    start = time.time()
    for i in range(1, total+1):
        raw = extract_text(pdf_bytes, i)
        text = normalize_text(refine_with_gemini(raw)).lower()
        match = None
        for n in name_targets:
            if normalize_text(n).lower() in text:
                match = n; break
        if not match:
            digits = re.sub(r"\D","",text)
            for a in account_targets:
                if re.sub(r"\D","",a) in digits:
                    match = a; break
        if match:
            matches.append({"page":i, "match":match})
        progress.progress(i/total)
    elapsed = time.time() - start
    st.success(f"プレビュー完了（{elapsed:.2f}s）")
    return matches

# 抽出処理
def run_extract():
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    progress = st.progress(0)
    zip_buf = io.BytesIO()
    start = time.time()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i in range(1, total+1):
            raw = extract_text(pdf_bytes, i)
            text = normalize_text(refine_with_gemini(raw)).lower()
            match = None
            for n in name_targets:
                if normalize_text(n).lower() in text:
                    match = n; break
            if not match:
                digits = re.sub(r"\D","",text)
                for a in account_targets:
                    if re.sub(r"\D","",a) in digits:
                        match = a; break
            if match:
                try:
                    writer = PdfWriter()
                    writer.add_page(reader.pages[i-1])
                    buf = io.BytesIO(); writer.write(buf)
                except Exception:
                    img = convert_from_bytes(pdf_bytes, first_page=i, last_page=i, dpi=300)[0]
                    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PDF")
                safe = re.sub(r"[\\/:*?\"<>|]","_",match)
                name = f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{i}.pdf"
                zf.writestr(name, buf.getvalue())
            progress.progress(i/total)
    zip_buf.seek(0)
    elapsed = time.time() - start
    st.success(f"抽出完了（{elapsed:.2f}s）")
    st.download_button(
        "抽出結果をZIPでダウンロード", zip_buf,
        file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip", mime="application/zip"
    )

# ボタン押下時実行
if preview_btn:
    preview_results = run_preview()
    if preview_results:
        st.subheader("マッチしたページ一覧")
        st.table(pd.DataFrame(preview_results))
    else:
        st.info("一致するページが見つかりませんでした。")

if extract_btn:
    run_extract()
