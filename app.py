import streamlit as st
import pandas as pd
import io
import zipfile
import re
from datetime import datetime
from pypdf import PdfReader
import pytesseract
# pdf2image may error if Poppler not installed
from pdf2image import convert_from_bytes, exceptions as pdf2image_exceptions
import google.generativeai as genai  # Gemini 用 SDKを正しいモジュールでインポート

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="OCR＋Gemini 支払通知書抽出", layout="wide")
st.title("OCR＋Gemini 支払通知書抽出ツール")

# ── Gemini クライアント初期化 ─────────────────────────
gemini_api_key = st.secrets["gemini"]["api_key"]
genai.configure(api_key=gemini_api_key)

# ── 正規化・補正関数 ─────────────────────────────────
def normalize_text(text: str) -> str:
    return "".join(text.split())

def refine_with_gemini(raw: str) -> str:
    prompt = f"以下のOCRテキストを日本語として正確に修正してください:\n```{raw}```"
    res = genai.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
        thinking_config=genai.ThinkingConfig(thinking_budget=1024)
    )
    return res.text

# ── アップロード ─────────────────────────────────────
pdf_file = st.file_uploader("PDFファイルをアップロード", type="pdf")
csv_file = st.file_uploader("CSVファイルをアップロード", type="csv")

# テキスト取得ユーティリティ

def extract_text_by_ocr_or_layer(pdf_bytes, page_num):
    try:
        # OCR 経由
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=300)
        raw = pytesseract.image_to_string(images[0], lang="jpn")
    except pdf2image_exceptions.PDFInfoNotInstalledError:
        # テキストレイヤー抽出
        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw = reader.pages[page_num-1].extract_text() or ""
    return raw

if pdf_file and csv_file:
    # --- CSV 読込 ---
    for enc in ("utf-8","cp932"):
        try:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, dtype=str, encoding=enc)
            st.success(f"CSV 読み込み成功（encoding={enc}）")
            break
        except Exception:
            st.warning(f"CSV 読み込み失敗（encoding={enc}）")
    else:
        st.error("CSV 読み込みできません。")
        st.stop()
    st.subheader("CSV プレビュー")
    st.dataframe(df)

    # --- ターゲット生成 ---
    name_targets = df.get("相手方", pd.Series()).dropna().astype(str).str.strip().tolist()
    account_targets = []
    for col in ("口座番号１","口座番号２","口座番号３"):
        if col in df.columns:
            account_targets += df[col].dropna().astype(str).str.strip().tolist()

    pdf_bytes = pdf_file.read()
    pdf_file.seek(0)

    # --- プレビュー ---
    if st.button("プレビュー実行"):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        st.write(f"▶ PDF 全ページ: {total} ページ")
        preview = []
        for i in range(1, total+1):
            raw = extract_text_by_ocr_or_layer(pdf_bytes, i)
            refined = refine_with_gemini(raw)
            text = normalize_text(refined).lower()
            match=None
            # 相手方優先
            for n in name_targets:
                if normalize_text(n).lower() in text:
                    match=n; break
            # 補助:口座番号
            if not match:
                digits = re.sub(r"\D","",text)
                for a in account_targets:
                    if re.sub(r"\D","",a) in digits:
                        match=a; break
            if match:
                preview.append({"page":i,"match":match})
        if preview:
            st.subheader("マッチ一覧")
            st.table(pd.DataFrame(preview))
        else:
            st.info("一致なし")

    # --- 抽出実行 ---
    if st.button("抽出実行"):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        zip_buf=io.BytesIO()
        with zipfile.ZipFile(zip_buf,"w") as zf:
            for i in range(1, total+1):
                raw = extract_text_by_ocr_or_layer(pdf_bytes,i)
                refined = refine_with_gemini(raw)
                text = normalize_text(refined).lower()
                match=None
                for n in name_targets:
                    if normalize_text(n).lower() in text:
                        match=n; break
                if not match:
                    digits = re.sub(r"\D","",text)
                    for a in account_targets:
                        if re.sub(r"\D","",a) in digits:
                            match=a; break
                if match:
                    # ページから抽出：テキストレイヤー優先
                    try:
                        # 直接PDFページ追加
                        page_writer = PdfWriter()
                        page_writer.add_page(reader.pages[i-1])
                        buf=io.BytesIO(); page_writer.write(buf)
                    except Exception:
                        # 画像→PDF
                        img = convert_from_bytes(pdf_bytes, first_page=i, last_page=i,dpi=300)[0]
                        buf=io.BytesIO(); img.convert("RGB").save(buf,format="PDF")
                    safe=re.sub(r"[\\/:*?\"<>|]","_",match)
                    name=f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{i}.pdf"
                    zf.writestr(name,buf.getvalue())
        zip_buf.seek(0)
        st.download_button("ZIPダウンロード",data=zip_buf,file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",mime="application/zip")
