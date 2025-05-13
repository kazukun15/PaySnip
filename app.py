import streamlit as st
import pandas as pd
import io
import zipfile
import re
from datetime import datetime
from pdf2image import convert_from_bytes
import pytesseract
from google import genai

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="OCR＋Gemini 支払通知書抽出", layout="wide")
st.title("OCR＋Gemini 支払通知書抽出ツール")

# ── Gemini クライアント初期化 ─────────────────────────
# secrets.toml に以下のように設定してください:
# [gemini]
# api_key = "YOUR_GEMINI_API_KEY"
gemini_api_key = st.secrets["gemini"]["api_key"]
genai_client = genai.Client(api_key=gemini_api_key)

# ── テキスト正規化・精緻化関数 ─────────────────────────
def normalize_text(text: str) -> str:
    return "".join(text.split())

def refine_with_gemini(raw_text: str) -> str:
    prompt = (
        "以下のOCRテキストを日本語として正確に修正してください:\n```" 
        + raw_text + "```"
    )
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=1024)
        )
    )
    return response.text

# ── ファイルアップロード ─────────────────────────────────
pdf_file = st.file_uploader("PDFファイルをアップロード", type="pdf")
csv_file = st.file_uploader("CSVファイルをアップロード", type="csv")

if pdf_file and csv_file:
    # --- CSV 読み込み（エンコーディング fallback） ---
    for enc in ("utf-8", "cp932"):
        try:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, dtype=str, encoding=enc)
            st.success(f"CSV読み込み成功（encoding={enc}）")
            break
        except Exception:
            st.warning(f"CSV読み込み失敗（encoding={enc}）")
    else:
        st.error("CSVの読み込みに失敗しました。別のエンコーディングを試してください。")
        st.stop()
    st.subheader("CSV プレビュー")
    st.dataframe(df)

    # --- 相手方・口座番号リスト作成 ---
    name_targets = df.get("相手方", pd.Series()).dropna().astype(str).str.strip().tolist()
    account_targets = []
    for col in ("口座番号１", "口座番号２", "口座番号３"):
        if col in df.columns:
            account_targets += df[col].dropna().astype(str).str.strip().tolist()

    # PDFバイト読み取り
    pdf_bytes = pdf_file.read()
    pdf_file.seek(0)

    # --- プレビュー実行 ---
    if st.button("プレビュー実行"):
        images = convert_from_bytes(pdf_bytes, dpi=300)
        preview = []
        for idx, img in enumerate(images, start=1):
            raw = pytesseract.image_to_string(img, lang="jpn")
            refined = refine_with_gemini(raw)
            text = normalize_text(refined)
            match = None
            # 相手方優先
            for name in name_targets:
                if normalize_text(name).lower() in text.lower():
                    match = name
                    break
            # 補助: 口座番号
            if not match:
                digits = re.sub(r"\D", "", text)
                for acc in account_targets:
                    if re.sub(r"\D", "", acc) in digits:
                        match = acc
                        break
            if match:
                preview.append({"page": idx, "match": match})
        if preview:
            st.subheader("マッチしたページ一覧")
            st.table(pd.DataFrame(preview))
        else:
            st.info("一致するページが見つかりませんでした。")

    # --- 抽出実行 & ダウンロード ---
    if st.button("抽出実行"):
        images = convert_from_bytes(pdf_bytes, dpi=300)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zipf:
            for idx, img in enumerate(images, start=1):
                raw = pytesseract.image_to_string(img, lang="jpn")
                refined = refine_with_gemini(raw)
                text = normalize_text(refined)
                match = None
                for name in name_targets:
                    if normalize_text(name).lower() in text.lower():
                        match = name
                        break
                if not match:
                    digits = re.sub(r"\D", "", text)
                    for acc in account_targets:
                        if re.sub(r"\D", "", acc) in digits:
                            match = acc
                            break
                if match:
                    # 画像 → PDF 化
                    pdf_img_buf = io.BytesIO()
                    img.convert("RGB").save(pdf_img_buf, format="PDF")
                    safe = re.sub(r"[\\/:*?\"<>|]", "_", match)
                    filename = f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{idx}.pdf"
                    zipf.writestr(filename, pdf_img_buf.getvalue())
        zip_buf.seek(0)
        st.download_button(
            label="抽出結果をZIPでダウンロード",
            data=zip_buf,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",
            mime="application/zip"
        )
