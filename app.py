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

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("📄 支払通知書抽出ツール")

# ── ヘルプ／使い方 ─────────────────────────────────
st.sidebar.header("🆘 使い方ガイド")
st.sidebar.markdown(
    """
    1. 左サイドバーからPDFとCSVを選択
    2. 自動でプレビューが表示されます
    3. 抽出ボタンを押して支払通知書を出力
    4. ZIP形式でダウンロードできます
    """
)

# ── ファイルアップロード ─────────────────────────
pdf_file = st.sidebar.file_uploader("📁 PDFファイル", type="pdf")
csv_file = st.sidebar.file_uploader("📁 CSVファイル", type="csv")

# ── セッションステートに保持 ─────────────────────
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
        st.sidebar.error(f"Gemini 初期化エラー: {e}")

# ── キャッシュ読み込み ─────────────────────────
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
        raise ValueError("PDFにページがありません。")
    return reader

# ── アップロードチェック ─────────────────────────
if 'pdf_bytes' not in st.session_state or 'csv_bytes' not in st.session_state:
    st.info("PDFとCSVをアップロードしてください。")
    st.stop()

# ── ファイルロード ─────────────────────────────────
try:
    df_csv = load_csv(st.session_state['csv_bytes'])
    reader = load_pdf(st.session_state['pdf_bytes'])
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()

# ── CSVプレビュー ─────────────────────────────────
st.subheader("CSVプレビュー")
st.dataframe(df_csv)

# ── 共通関数 ─────────────────────────────────────────
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)

# ── マッチング処理────────────────────────────────────
def match_pages(reader, names_map, accounts_map):
    results = []
    total = len(reader.pages)
    progress = st.progress(0)
    for idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        norm = normalize_text(raw)
        matched = None
        # 名前優先
        for k,v in names_map.items():
            if k in norm:
                matched = v
                break
        # 口座番号
        if not matched:
            digits = re.sub(r"\D", "", norm)
            for k,v in accounts_map.items():
                if k in digits:
                    matched = v
                    break
        if matched:
            results.append({"page": idx, "match": matched})
        progress.progress(idx/total)
    return results

# ── マップ生成────────────────────────────────────────
raw_names = df_csv['相手方'].dropna().tolist()
names_map = {normalize_text(n): n for n in raw_names}
raw_accounts = []
for col in ['口座番号１','口座番号２','口座番号３']:
    raw_accounts += df_csv.get(col, pd.Series(dtype=str)).dropna().tolist()
accounts_map = {re.sub(r"\D","",a):a for a in raw_accounts}

# ── プレビュー────────────────────────────────────────
st.subheader("プレビュー：マッチング結果")
preview = match_pages(reader, names_map, accounts_map)
if preview:
    st.table(preview)
else:
    st.warning("一致するページがありませんでした。")

# ── 抽出ボタン───────────────────────────────────────
if st.button("抽出実行", use_container_width=True):
    start = time.time()
    matches = preview
    if not matches:
        st.error("抽出対象がありません。")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf,'w') as zf:
            for item in matches:
                pg = item['page']
                key = item['match']
                writer = PdfWriter()
                writer.add_page(reader.pages[pg-1])
                fbuf = io.BytesIO()
                writer.write(fbuf)
                fname = f"{datetime.now():%Y%m%d}_支払通知書_{key}_p{pg}.pdf"
                zf.writestr(fname, fbuf.getvalue())
        buf.seek(0)
        st.download_button(
            "ZIPダウンロード", data=buf,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip",
            mime="application/zip"
        )
        st.success(f"完了: {len(matches)} 件 ({time.time()-start:.2f}秒)")
