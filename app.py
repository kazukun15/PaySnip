import streamlit as st
import pandas as pd
from pypdf import PdfReader, PdfWriter
import io
import zipfile
from datetime import datetime
import re

st.set_page_config(page_title="支払通知書 ページ抽出ツール", layout="wide")
st.title("支払通知書 ページ抽出ツール")

# --- 1. ファイルアップロード ---
pdf_file = st.file_uploader("PDFファイルをアップロード", type="pdf")
csv_file = st.file_uploader("CSVファイルをアップロード", type="csv")

# テキスト正規化関数
def normalize_text(text: str) -> str:
    return ''.join(text.split())  # 空白や改行を除去
def normalize_digits(text: str) -> str:
    return re.sub(r"\D", "", text)  # 数字以外を除去

# PDFページからマッチを検出する関数
def find_matches(pdf_stream, targets):
    reader = PdfReader(pdf_stream)
    matches = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text_norm = normalize_text(raw)
        digits_norm = normalize_digits(raw)
        for t in targets:
            t_norm = normalize_text(t)
            # 数字を含むターゲットは数字正規化テキストでマッチ
            if re.search(r"\d", t):
                if normalize_digits(t) in digits_norm:
                    matches.append({"page": i, "match": t})
                    break
            else:
                # 名前など文字ターゲットは空白除去テキストでマッチ（大文字小文字同一視）
                if t_norm.lower() in text_norm.lower():
                    matches.append({"page": i, "match": t})
                    break
    return matches

if pdf_file and csv_file:
    # --- 2. CSV 読み込み（エンコーディング fallback） ---
    for enc in ("utf-8", "cp932"):
        try:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, dtype=str, encoding=enc)
            st.success(f"CSV 読み込み成功（encoding={enc}）")
            break
        except UnicodeDecodeError:
            st.warning(f"CSV 読み込みに失敗しました（encoding={enc}）")
    else:
        st.error("CSV の読み込みに失敗しました。別のエンコーディングを試してください。")
        st.stop()

    st.subheader("CSV プレビュー")
    st.dataframe(df)

    # --- 3. マッチ対象文字列の抽出 ---
    targets = set()
    for col in ["相手方", "口座番号１", "口座番号２", "口座番号３"]:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            targets |= set(values)

    # プレビュー
    if st.button("プレビュー実行"):
        matches = find_matches(pdf_file, targets)
        if matches:
            preview_df = pd.DataFrame(matches)
            st.subheader("マッチしたページ一覧")
            st.dataframe(preview_df)
        else:
            st.info("一致するページが見つかりませんでした。")

    # 抽出実行＆ZIPダウンロード
    if st.button("抽出実行"):
        matches = find_matches(pdf_file, targets)
        if not matches:
            st.error("一致するページがないため、抽出できません。")
            st.stop()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            reader = PdfReader(pdf_file)
            for item in matches:
                page_no = item["page"]
                match_str = item["match"]
                writer = PdfWriter()
                writer.add_page(reader.pages[page_no - 1])
                pdf_bytes = io.BytesIO()
                writer.write(pdf_bytes)
                name = f"{datetime.now():%Y%m%d}_支払通知書_{match_str}_p{page_no}.pdf"
                zipf.writestr(name, pdf_bytes.getvalue())

        zip_buffer.seek(0)
        st.download_button(
            label="抽出結果をZIPでダウンロード",
            data=zip_buffer,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",
            mime="application/zip"
        )
