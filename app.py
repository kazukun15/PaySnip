import streamlit as st
import pandas as pd
import io
import zipfile
import re
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pypdf import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import InternalServerError

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("📄 支払通知書抽出ツール")

# ── サイドバー：ファイル選択・設定 ─────────────────────────
st.sidebar.header("1. ファイルをアップロード")
pdf_file = st.sidebar.file_uploader("PDFファイル (.pdf)", type="pdf")
csv_file = st.sidebar.file_uploader("CSVファイル (.csv)", type="csv")

st.sidebar.header("2. オプション")
enable_refine = st.sidebar.checkbox("Gemini 補正を有効にする", value=False)

action_preview = st.sidebar.button("プレビュー実行")
action_extract = st.sidebar.button("抽出実行")

# ── Gemini モデル初期化 ─────────────────────────────────
@st.cache_resource
def init_gemini_model(api_key: str) -> Optional[genai.GenerativeModel]:
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    except Exception:
        return None

gemini_api_key = st.secrets.get("gemini", {}).get("api_key", "")
model = init_gemini_model(gemini_api_key)

# ── ユーティリティ関数 ─────────────────────────────────
def normalize_text(text: str) -> str:
    """空白除去して比較しやすくする"""
    return re.sub(r"\s+", "", text)

def refine_text(raw: str, page: int) -> str:
    """Gemini APIで誤りを補正（APIエラーなら生テキスト返却）"""
    if not (model and enable_refine):
        return raw
    try:
        # プロンプトにバックティックを含めず、シンプルな文章形式に変更してf-stringエラー回避
        prompt = f"""PDFの{page}ページから抽出された支払通知書のテキストを、誤字脱字なく自然な日本語に修正してください。

{raw}
"""
        res = model.generate_content(prompt)
        return res.text
    except InternalServerError:
        return raw
    except Exception:
        return raw

# ── マッチング処理 ─────────────────────────────────
def find_matches(
    reader: PdfReader,
    names: List[str],
    accounts: List[str],
) -> List[Dict]:
    """テキストレイヤー → 補正 → 名前 or 口座番号でマッチ"""
    results = []
    total = len(reader.pages)
    for idx in range(total):
        raw = reader.pages[idx].extract_text() or ""
        text = raw
        # 補正テキスト
        if enable_refine:
            text = refine_text(raw, idx+1)
        norm = normalize_text(text)
        matched: Optional[str] = None
        # 名前優先
        for name in names:
            if normalize_text(name) in norm:
                matched = name
                break
        # 口座番号補助
        if not matched:
            digits = re.sub(r"\D", "", text)
            for acc in accounts:
                if re.sub(r"\D", "", acc) in digits:
                    matched = acc
                    break
        if matched:
            results.append({"page": idx+1, "match": matched})
    return results

# ── アプリ本体 ─────────────────────────────────────────
if not pdf_file or not csv_file:
    st.warning("PDFとCSVを両方アップロードしてください。")
    st.stop()

# データ読み込み
csv_df = load_csv(csv_file)
pdf_reader = load_pdf_reader(pdf_file)
names = csv_df.get("相手方", pd.Series()).dropna().str.strip().tolist()
accounts = sum([csv_df.get(c, pd.Series()).dropna().str.strip().tolist()
                for c in ["口座番号１","口座番号２","口座番号３"]], [])

st.subheader("CSV プレビュー")
st.dataframe(csv_df.head(5))
st.write(f"アップロードPDFページ数: {len(pdf_reader.pages)} ページ")

# プレビュー
if action_preview:
    with st.spinner("プレビュー実行中…"):
        t0 = time.time()
        preview = find_matches(pdf_reader, names, accounts)
        dt = time.time() - t0
    st.success(f"プレビュー完了 ({dt:.2f}秒)")
    if preview:
        st.table(pd.DataFrame(preview))
    else:
        st.warning("一致するページがありませんでした。")

# 抽出
if action_extract:
    with st.spinner("抽出実行中…"):
        t0 = time.time()
        matches = find_matches(pdf_reader, names, accounts)
        # ZIP作成
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for m in matches:
                page = m['page'] - 1
                writer = PdfWriter()
                writer.add_page(pdf_reader.pages[page])
                b = io.BytesIO()
                writer.write(b)
                safe = re.sub(r"[\\/:*?\"<>|]", "_", m['match'])
                fname = f"{datetime.now():%Y%m%d}_支払通知書_{safe}_p{m['page']}.pdf"
                zf.writestr(fname, b.getvalue())
        buf.seek(0)
        dt = time.time() - t0
    if matches:
        st.success(f"抽出完了 ({dt:.2f}秒) - {len(matches)}ページをZIP化しました。")
        st.download_button(
            "ZIPダウンロード", buf,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書.zip"
        )
        st.subheader("抽出結果詳細")
        st.dataframe(pd.DataFrame(matches))
    else:
        st.warning(f"抽出対象のページが見つかりませんでした。 ({dt:.2f}秒)")
