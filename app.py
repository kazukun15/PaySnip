import streamlit as st
import pandas as pd
import io
import zipfile
import re
import time
from datetime import datetime
from pypdf import PdfReader, PdfWriter
import fitz  # PyMuPDF

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("支払通知書抽出ツール")

# ── サイドバー：設定・アップロード ─────────────────────────
st.sidebar.header("ファイル選択")
pdf_file = st.sidebar.file_uploader("PDFファイル", type="pdf")
csv_file = st.sidebar.file_uploader("CSVファイル", type="csv")
preview_btn = st.sidebar.button("プレビュー実行")
extract_btn = st.sidebar.button("抽出実行")

# ── OCR不要: PyMuPDFで名前検出 ─────────────────────────
def extract_name_from_page(page) -> str:
    # テキストブロックをY座標順に取得
    blocks = sorted(page.get_text('blocks'), key=lambda b: b[1])
    for b in blocks:
        text = b[4].strip()
        if text.endswith('様'):
            # 例: '町村議会議員共済会 様' -> '町村議会議員共済会'
            return text[:-1].strip()
    return ''

# ── CSV読み込み ─────────────────────────────────
def load_csv(uploaded_file):
    uploaded_file.seek(0)
    # 全行を読み込む
    df = pd.read_csv(uploaded_file, dtype=str, encoding='utf-8', engine='python')
    df = df.fillna('')
    # 正規化マップ
    name_map = {re.sub(r'\s+', '', name): name for name in df['相手方'].tolist() if name}
    acct_list = []
    for col in ['口座番号１','口座番号２','口座番号３']:
        acct_list += df[col].dropna().tolist()
    acct_map = {re.sub(r'\D','',acc): acc for acc in acct_list if acc}
    return df, name_map, acct_map

# ── PDFリーダー（キャッシュ無効化）─────────────────────────
def get_pdf_reader(uploaded_file) -> PdfReader:
    uploaded_file.seek(0)
    return PdfReader(io.BytesIO(uploaded_file.read()))

if not pdf_file or not csv_file:
    st.info("サイドバーからPDFとCSVをアップロードしてください。")
    st.stop()

# データ準備
csv_df, name_map, acct_map = load_csv(csv_file)
st.subheader("CSV プレビュー (先頭5行)")
st.dataframe(csv_df.head())
pdf_reader = get_pdf_reader(pdf_file)
st.write(f"PDF: {pdf_file.name}, {len(pdf_reader.pages)} ページ")

# マッチング関数

def match_pages(reader, name_map, acct_map, use_preview=False):
    results = []
    for i, page in enumerate(reader.pages, start=1):
        text_name = extract_name_from_page(page)
        matched = ''
        if text_name:
            key = re.sub(r'\s+','', text_name)
            if key in name_map:
                matched = name_map[key]
        if not matched:
            # ページ全体をテキストで取得して数字抽出
            txt = page.get_text()
            digits = re.sub(r'\D','', txt)
            for ak, ov in acct_map.items():
                if ak and ak in digits:
                    matched = ov
                    break
        if matched:
            results.append({'page': i, 'match': matched})
    return results

# プレビュー
if preview_btn:
    st.subheader("プレビュー結果")
    with st.spinner("プレビュー中…"):
        preview = match_pages(pdf_reader, name_map, acct_map, use_preview=True)
    if preview:
        st.table(pd.DataFrame(preview))
    else:
        st.warning("一致ページなし。")

# 抽出
if extract_btn:
    st.subheader("抽出結果")
    with st.spinner("抽出中…"):
        matched = match_pages(pdf_reader, name_map, acct_map)
    if not matched:
        st.warning("対象ページが見つかりませんでした。")
        st.stop()
    # ZIP作成
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item in matched:
            idx = item['page'] - 1
            writer = PdfWriter()
            writer.add_page(pdf_reader.pages[idx])
            bio = io.BytesIO()
            writer.write(bio)
            bio.seek(0)
            name_safe = re.sub(r'[\\/*?:"<>|]','_', item['match'])
            fname = f"{datetime.now():%Y%m%d}_支払通知書_{name_safe}_p{item['page']}.pdf"
            zf.writestr(fname, bio.read())
    buf.seek(0)
    st.download_button("ZIPダウンロード", buf, file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip", mime='application/zip')
