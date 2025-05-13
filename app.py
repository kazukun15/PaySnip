import streamlit as st
import pandas as pd
from pypdf import PdfReader, PdfWriter
import io
import zipfile
from datetime import datetime

st.set_page_config(page_title="支払通知書 ページ抽出ツール", layout="wide")
st.title("支払通知書 ページ抽出ツール")

# --- 1. ファイルアップロード ---
pdf_file = st.file_uploader("PDFファイルをアップロード", type="pdf")
csv_file = st.file_uploader("CSVファイルをアップロード", type="csv")

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

    # 内部関数：PDF を読み込んでマッチリストを返す
    def find_matches(pdf_stream, targets):
        reader = PdfReader(pdf_stream)
        matches = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for t in targets:
                if t in text:
                    matches.append({"page": i, "match": t})
                    break
        return matches

    # --- 4. プレビュー表示 ---
    if st.button("プレビュー実行"):
        matches = find_matches(pdf_file, targets)
        if matches:
            preview_df = pd.DataFrame(matches)
            st.subheader("マッチしたページ一覧")
            st.dataframe(preview_df)
        else:
            st.info("一致するページが見つかりませんでした。")

    # --- 5. 抽出実行＆ZIP ダウンロード ---
    if st.button("抽出実行"):
        matches = find_matches(pdf_file, targets)
        if not matches:
            st.error("一致するページがないため、抽出できません。")
            st.stop()

        # ZIP をメモリ上で作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            reader = PdfReader(pdf_file)  # 再度読み込み
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
