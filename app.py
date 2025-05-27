import streamlit as st
import pandas as pd
import io
import zipfile
import re
import time
from datetime import datetime
from pypdf import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import InternalServerError
from st_aggrid import AgGrid, GridOptionsBuilder

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("📄 支払通知書抽出ツール")

# ── ヘルプ／使い方 ─────────────────────────────────
st.sidebar.header("🆘 使い方ガイド")
st.sidebar.markdown(
    """
    1. 左サイドバーからPDFとCSVを選択
    2. 自動でプレビューが表示されます
    3. エラーがなければ抽出ボタンを押してください
    4. ZIP形式で支払通知書をダウンロードできます
    """
)

# ── サイドバー：ファイルアップロード ─────────────────────────
pdf_file = st.sidebar.file_uploader("📁 PDFファイルをアップロード", type="pdf", key="pdf_uploader")
csv_file = st.sidebar.file_uploader("📁 CSVファイルをアップロード", type="csv", key="csv_uploader")

# ── セッションステートにファイルを保持 ─────────────────────
if 'pdf_bytes' not in st.session_state and pdf_file:
    st.session_state['pdf_bytes'] = pdf_file.read()
if 'csv_bytes' not in st.session_state and csv_file:
    st.session_state['csv_bytes'] = csv_file.read()

# ── Google Gemini 初期化 ─────────────────────────
gemini_api_key = st.secrets.get("gemini_api_key", "")
model = None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.sidebar.error(f"Gemini初期化エラー: {e}")

# ── キャッシュ付き読み込み関数 ─────────────────────────
@st.cache_data
def load_csv(bytes_data: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "cp932", "shift_jis"):  
        try:
            return pd.read_csv(io.BytesIO(bytes_data), dtype=str, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV読み込みに失敗しました。エンコーディングを確認してください。")

@st.cache_data
def load_pdf(bytes_data: bytes) -> PdfReader:
    reader = PdfReader(io.BytesIO(bytes_data))
    if not reader.pages:
        raise ValueError("PDFにページが含まれていません。")
    return reader

# ── 入力ファイルチェック ─────────────────────────────────
if 'pdf_bytes' not in st.session_state or 'csv_bytes' not in st.session_state:
    st.info("PDFとCSVをアップロードすると自動でプレビューします。")
    st.stop()

# ── ファイルロード ─────────────────────────────────────
try:
    df_csv = load_csv(st.session_state['csv_bytes'])
    reader = load_pdf(st.session_state['pdf_bytes'])
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()

# ── CSVプレビュー（AgGrid）──────────────────────────────
st.subheader("CSVデータプレビュー")
gb = GridOptionsBuilder.from_dataframe(df_csv)
gb.configure_pagination(paginationAutoPageSize=True)
gb.configure_default_column(filterable=True, sortable=True)
AgGrid(df_csv, gridOptions=gb.build(), height=300)

# ── 共通関数 ─────────────────────────────────────────
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)

# モデル補正関数省略（必要に応じ再定義）

# ── マッチング処理────────────────────────────────────
def match_pages(reader, names_map, accounts_map, use_refine=False):
    results = []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        st.progress(i/total)
        raw = page.extract_text() or ""
        norm = normalize_text(raw)
        # 名前優先
        matched = None
        for k,v in names_map.items():
            if k in norm:
                matched = v
                break
        if not matched:
            digits = re.sub(r"\D", "", norm)
            for k,v in accounts_map.items():
                if k in digits:
                    matched = v
                    break
        if matched:
            results.append((i, matched))
    return results

# ── マップ生成────────────────────────────────────────
raw_names = df_csv['相手方'].dropna().tolist()
names_map = {normalize_text(n): n for n in raw_names}
raw_accounts = []
for col in ['口座番号１','口座番号２','口座番号３']:
    raw_accounts += df_csv.get(col, pd.Series()).dropna().tolist()
accounts_map = {re.sub(r"\D","",a):a for a in raw_accounts}

# ── プレビュー結果表示────────────────────────────────
st.subheader("プレビュー結果（自動マッチング）")
preview = match_pages(reader, names_map, accounts_map)
if preview:
    st.table(pd.DataFrame(preview, columns=['ページ','マッチ']))
else:
    st.warning("一致するページが見つかりませんでした。")

# ── 抽出ボタン───────────────────────────────────────
if st.button("抽出実行", use_container_width=True):
    start = time.time()
    matches = match_pages(reader, names_map, accounts_map)
    if not matches:
        st.error("抽出対象がありません。")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf,'w') as zf:
            for pg, key in matches:
                writer = PdfWriter()
                writer.add_page(reader.pages[pg-1])
                fbuf = io.BytesIO()
                writer.write(fbuf)
                fname = f"{datetime.now():%Y%m%d}_支払通知書_{key}_p{pg}.pdf"
                zf.writestr(fname, fbuf.getvalue())
        buf.seek(0)
        st.download_button("ZIPダウンロード", data=buf,
                           file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",
                           mime="application/zip")
        st.success(f"完了: {len(matches)} 件 ({time.time()-start:.2f}秒)")
