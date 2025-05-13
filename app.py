import streamlit as st
import pandas as pd
import io
import zipfile
import re
from datetime import datetime
from pypdf import PdfReader, PdfWriter
import pytesseract
from pdf2image import convert_from_bytes, exceptions as pdf2image_exceptions
import google.generativeai as genai
from google.api_core.exceptions import InternalServerError

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="OCR＋Gemini 支払通知書抽出", layout="wide")
st.title("OCR＋Gemini 支払通知書抽出ツール")

# ── Gemini SDK 初期化 ─────────────────────────────────
gemini_api_key = st.secrets["gemini"]["api_key"]
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

# ── 正規化・補正関数 ─────────────────────────────────
def normalize_text(text: str) -> str:
    return "".join(text.split())

def refine_with_gemini(raw: str) -> str:
    """
    Gemini でテキストを精緻化。内部サーバーエラー時はOCR原文を返す。
    """
    try:
        prompt = f"以下のOCRテキストを日本語として正確に修正してください:\n```{raw}```"
        response = model.generate_content(prompt)
        return response.text
    except InternalServerError as e:
        # Geminiサーバーエラー発生時はログ出力し、生のOCRテキストを返す
        st.warning(f"Gemini補正エラー: {e}")
        return raw
    except Exception as e:
        st.warning(f"Gemini補正で予期しないエラー: {e}")
        return raw

# ── テキスト取得ユーティリティ ───────────────────────────
def extract_text_by_ocr_or_layer(pdf_bytes: bytes, page_num: int) -> str:
    try:
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=300)
        raw = pytesseract.image_to_string(images[0], lang="jpn")
    except pdf2image_exceptions.PDFInfoNotInstalledError:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw = reader.pages[page_num - 1].extract_text() or ""
    except Exception as e:
        # その他のエラー時もテキストレイヤーでフォールバック
        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw = reader.pages[page_num - 1].extract_text() or ""
    return raw

# ── ファイルアップロード ─────────────────────────────────
pdf_file = st.file_uploader("PDFファイルをアップロード", type="pdf")
csv_file = st.file_uploader("CSVファイルをアップロード", type="csv")

if pdf_file and csv_file:
    # --- CSV読み込み（エンコーディング fallback） ---
    for enc in ("utf-8", "cp932"):
        try:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, dtype=str, encoding=enc)
            st.success(f"CSV読み込み成功（encoding={enc}）")
            break
        except Exception:
            st.warning(f"CSV読み込み失敗（encoding={enc}）")
    else:
        st.error("CSV読み込みに失敗しました。別のエンコーディングを試してください。")
        st.stop()
    st.subheader("CSV プレビュー")
    st.dataframe(df)

    # --- ターゲット作成 ---
    name_targets = df.get("相手方", pd.Series()).dropna().astype(str).str.strip().tolist()
    account_targets = []
    for col in ("口座番号１", "口座番号２", "口座番号３"):
        if col in df.columns:
            account_targets += df[col].dropna().astype(str).str.strip().tolist()

    # PDF bytes
    pdf_bytes = pdf_file.read()
    pdf_file.seek(0)

    # --- プレビュー実行 ---
    if st.button("プレビュー実行"):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        st.write(f"▶ PDF 全ページ: {total} ページ")
        preview = []
        for i in range(1, total + 1):
            raw = extract_text_by_ocr_or_layer(pdf_bytes, i)
            refined = refine_with_gemini(raw)
            text = normalize_text(refined).lower()
            match = None
            # 相手方優先
            for n in name_targets:
                if normalize_text(n).lower() in text:
                    match = n
                    break
            # 補助: 口座番号
            if not match:
                digits = re.sub(r"\D", "", text)
                for a in account_targets:
                    if re.sub(r"\D", "", a) in digits:
                        match = a
                        break
            if match:
                preview.append({"page": i, "match": match})
        if preview:
            st.subheader("マッチしたページ一覧")
            st.table(pd.DataFrame(preview))
        else:
            st.info("一致するページが見つかりませんでした。")

    # --- 抽出実行 & ZIPダウンロード ---
    if st.button("抽出実行"):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = len(reader.pages)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i in range(1, total + 1):
                raw = extract_text_by_ocr_or_layer(pdf_bytes, i)
                refined = refine_with_gemini(raw)
                text = normalize_text(refined).lower()
                match = None
                for n in name_targets:
                    if normalize_text(n).lower() in text:
                        match = n
                        break
                if not match:
                    digits = re.sub(r"\D", "", text)
                    for a in account_targets:
                        if re.sub(r"\D", "", a) in digits:
                            match = a
                            break
                if match:
                    # テキストレイヤー優先でPDF化
                    try:
                        writer = PdfWriter()
                        writer.add_page(reader.pages[i - 1])
                        buf = io.BytesIO()
                        writer.write(buf)
                    except Exception:
                        img = convert_from_bytes(pdf_bytes, first_page=i, last_page=i, dpi=300)[0]
                        buf = io.BytesIO()
                        img.convert("RGB").save(buf, format="PDF")
                    safe = re.sub(r"[\\/:*?\"<>|]", "_", match)
                    filename = f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{i}.pdf"
                    zf.writestr(filename, buf.getvalue())
        zip_buf.seek(0)
        st.download_button(
            label="抽出結果をZIPでダウンロード",
            data=zip_buf,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",
            mime="application/zip"
        )
