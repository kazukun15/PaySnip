import streamlit as st
import pandas as pd
import io
import zipfile
import re
import time
from datetime import datetime
from pypdf import PdfWriter
import fitz  # PyMuPDF for reliable text extraction

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("📄 支払通知書抽出ツール")

# ── サイドバー：使い方 ─────────────────────────────────
st.sidebar.header("🆘 使い方ガイド")
st.sidebar.markdown(
    """
    1. 左サイドバーからPDFとCSVを選択
    2. プレビュー結果が表示されるので確認
    3. 【抽出実行】を押してZIP形式でダウンロード
    """
)

# ── ファイルアップロード ─────────────────────────
pdf_file = st.sidebar.file_uploader("📁 PDFファイルを選択", type="pdf")
csv_file = st.sidebar.file_uploader("📁 CSVファイルを選択", type="csv")

# ── アップロードチェック ─────────────────────────
if not pdf_file or not csv_file:
    st.sidebar.info("PDFとCSVをアップロードしてください。")
    st.stop()

# ── ファイル読み込みと可視化 ─────────────────────────
@st.cache_data
def load_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSVを読み込めません。エンコーディングを確認してください。")

@st.cache_data
def load_docs(pdf_bytes: bytes):
    # PyPDF と PyMuPDF の両方で読み込み
    pdf_reader = fitz.open(stream=pdf_bytes, filetype="pdf")
    return pdf_reader

# セッションストレージ
if 'pdf_bytes' not in st.session_state:
    st.session_state['pdf_bytes'] = pdf_file.read()
if 'csv_bytes' not in st.session_state:
    st.session_state['csv_bytes'] = csv_file.read()

# 読み込み
try:
    df_csv = load_csv(st.session_state['csv_bytes'])
    fitz_doc = load_docs(st.session_state['pdf_bytes'])
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()

# ── CSVプレビュー ─────────────────────────────────
st.subheader("📋 CSVプレビュー")
st.dataframe(df_csv)

# ── 正規化関数 ─────────────────────────────────────
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", s)

# ── 名称マップ生成 ─────────────────────────────────
raw_names = df_csv.get('相手方', pd.Series(dtype=str)).dropna().tolist()
names_map = {normalize_text(n): n for n in raw_names}

# ── 口座番号マップ生成 ─────────────────────────────────
raw_acc = []
for col in ['口座番号１', '口座番号２', '口座番号３']:
    raw_acc += df_csv.get(col, pd.Series(dtype=str)).dropna().tolist()
accounts_map = {re.sub(r"\D", "", a): a for a in raw_acc if re.sub(r"\D", "", a)}

# ── ページマッチング関数 ─────────────────────────────
def match_pages(fitz_doc, names_map, accounts_map):
    results = []
    total = fitz_doc.page_count
    progress = st.progress(0)

    for i in range(total):
        page = fitz_doc.load_page(i)
        blocks = page.get_text('blocks')  # (x0, y0, x1, y1, text, block_no)
        matched = None
        # 1. ブロック内の「様」で名前抽出・照合
        for b in sorted(blocks, key=lambda x: x[1]):  # y0順
            text = b[4]
            if '様' in text:
                # 最初に現れるテキストから「様」前の文字列を取得
                m = re.search(r'([^\s].+?)様', text)
                if m:
                    name = normalize_text(m.group(1))
                    if name in names_map:
                        matched = names_map[name]
                        break
        # 2. 名前未一致なら口座番号照合
        if not matched:
            full_text = normalize_text(page.get_text())
            digits = re.sub(r"\D", "", full_text)
            for acc_norm, acc_orig in accounts_map.items():
                if acc_norm and acc_norm in digits:
                    matched = acc_orig
                    break
        if matched:
            results.append({'page': i+1, 'match': matched})
        progress.progress((i+1)/total)
    return results

# ── プレビュー結果 ─────────────────────────────────
st.subheader("🔍 プレビュー：マッチング結果")
preview = match_pages(fitz_doc, names_map, accounts_map)
if preview:
    st.table(preview)
else:
    st.warning("一致するページが見つかりませんでした。")

# ── 抽出・ZIP化 ─────────────────────────────────────
if st.button("🚀 抽出実行", use_container_width=True):
    if not preview:
        st.error("抽出対象がありません。")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for item in preview:
                pg = item['page']
                key = item['match']
                writer = PdfWriter()
                src = fitz_doc.load_page(pg-1).get_pdf_xref()
                # PyMuPDFページをPyPDFWriterに追加
                pix = fitz_doc.load_page(pg-1).get_pixmap()
                # PDF単ページ作成（ここは既存のpypdfを継続利用）
                writer.add_page(fitz_doc._get_page(pg-1))
                fbuf = io.BytesIO()
                writer.write(fbuf)
                name_safe = re.sub(r'[\\/*?:"<>|]', '_', key)
                fname = f"{datetime.now():%Y%m%d}_支払通知書_{name_safe}_p{pg}.pdf"
                zf.writestr(fname, fbuf.getvalue())
        buf.seek(0)
        st.download_button(
            "📥 ZIPダウンロード", data=buf,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",
            mime="application/zip"
        )
        st.success(f"完了: {len(preview)} 件 ({time.time()-start:.2f}秒)")
