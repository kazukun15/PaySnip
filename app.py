import streamlit as st
import pandas as pd
from pypdf import PdfReader, PdfWriter
from pdfminer.high_level import extract_text
import io, zipfile
from datetime import datetime

st.title("支払通知書 ページ抽出ツール")

# 1. アップローダー
pdf_file = st.file_uploader("PDFファイルをアップロード", type="pdf")  # :contentReference[oaicite:6]{index=6}
csv_file = st.file_uploader("CSVファイルをアップロード", type="csv")  # :contentReference[oaicite:7]{index=7}

if pdf_file and csv_file:
    # 2. CSV 読み込み
    df = pd.read_csv(csv_file, dtype=str)  # :contentReference[oaicite:8]{index=8}
    st.subheader("CSV プレビュー")
    st.dataframe(df)  # :contentReference[oaicite:9]{index=9}

    # 3. ターゲット文字列セット
    targets = set()
    for col in ["相手方","口座番号１","口座番号２","口座番号３"]:
        if col in df.columns:
            targets |= set(df[col].dropna().astype(str).str.strip())

    # プレビュー用ボタン
    if st.button("プレビュー実行"):
        reader = PdfReader(pdf_file)  # :contentReference[oaicite:10]{index=10}
        matches = []
        # ページごとにテキスト抽出 → マッチ判定
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""  # :contentReference[oaicite:11]{index=11}
            for t in targets:
                if t in text:
                    matches.append({"page": i, "match": t})
                    break
        preview_df = pd.DataFrame(matches)
        st.subheader("マッチしたページ一覧")
        st.dataframe(preview_df)  # :contentReference[oaicite:12]{index=12}

    # 4. 実行ボタン
    if st.button("抽出実行"):
        reader = PdfReader(pdf_file)  # :contentReference[oaicite:13]{index=13}
        # メモリ上 ZIP 作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:  # :contentReference[oaicite:14]{index=14}
            for item in matches:
                page_no = item["page"]
                match_str = item["match"]
                writer = PdfWriter()
                writer.add_page(reader.pages[page_no-1])  # :contentReference[oaicite:15]{index=15}
                pdf_bytes = io.BytesIO()
                writer.write(pdf_bytes)
                name = f"{datetime.now().strftime('%Y%m%d')}_支払通知書_{match_str}_p{page_no}.pdf"
                zipf.writestr(name, pdf_bytes.getvalue())

        zip_buffer.seek(0)
        st.download_button(
            "ZIP をダウンロード",
            data=zip_buffer,
            file_name=f"{datetime.now().strftime('%Y%m%d')}_支払通知書.zip",
            mime="application/zip"
        )  # :contentReference[oaicite:16]{index=16}
