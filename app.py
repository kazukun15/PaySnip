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
from typing import List, Dict, Optional, Tuple

# ── Streamlit 設定 ─────────────────────────────────
st.set_page_config(page_title="支払通知書抽出ツール", layout="wide")
st.title("支払通知書抽出ツール")

# ── サイドバー：設定・アップロード (APIキーはsecretsから自動取得) ─────────────────────────
st.sidebar.header("ファイル選択")
pdf_file = st.sidebar.file_uploader("PDFファイル", type="pdf")
csv_file = st.sidebar.file_uploader("CSVファイル", type="csv")
preview_btn = st.sidebar.button("プレビュー実行")
extract_btn = st.sidebar.button("抽出実行")

# ── Gemini SDK 初期化 (APIキーは secrets.toml で設定) ─────────────────────────
gemini_api_key = st.secrets.get("gemini", {}).get("api_key")
model = None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    # model = genai.GenerativeModel('gemini-1.5-flash-latest') # モデル名を最新版に更新 (適宜確認してください)
    # 最新のモデル名を確認してください。提供されたコードでは 'gemini-2.5-flash-preview-04-17' でしたが、
    # 利用可能なモデルリストから適切なものを選択してください。
    # 一般的に 'gemini-1.5-flash-latest' や 'gemini-1.5-pro-latest' などが利用可能です。
    # ここでは例として 'gemini-1.5-flash-latest' をコメントアウトで示しています。
    # 使用するモデルによっては、APIの互換性や機能が異なる場合があるため、ドキュメントを参照してください。
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # 例: gemini-1.5-flash
    except Exception as e:
        st.error(f"Geminiモデルの初期化に失敗しました: {e}")
        model = None


# ── 共通関数 ─────────────────────────────────────
def normalize_text(text: str) -> str:
    return "".join(text.split())

def refine_text_with_gemini(raw_text: str, page_num: int) -> Tuple[str, bool]:
    """
    Gemini APIを使用してテキストを修正します。
    エラーが発生した場合は元のテキストとエラーフラグを返します。
    """
    if not model:
        return raw_text, False
    try:
        # プロンプトをより具体的に、かつ簡潔にすることで、期待する結果を得やすくし、処理時間を短縮できる可能性があります。
        # 例：「以下のOCR抽出テキストを、誤字脱字を修正し、自然な日本語の文章にしてください。特に固有名詞や数値の正確性を重視してください。」
        prompt = f"以下のテキストはPDFの {page_num} ページから抽出された支払通知書の一部です。これを正確な日本語のテキストに修正してください:\n\n```{raw_text}```\n\n修正後のテキスト:"
        response = model.generate_content(
            prompt,
            # generation_config=genai.types.GenerationConfig( # 必要に応じて設定
            #     temperature=0.2, # 低めに設定してより決定的な出力を促す
            # )
            )
        return response.text, False
    except InternalServerError as e:
        st.warning(f"ページ {page_num} のテキスト補正中にAPI内部エラーが発生しました。元のテキストを使用します。エラー: {e}")
        return raw_text, True
    except Exception as e: # 他のAPI関連エラーもキャッチ
        st.warning(f"ページ {page_num} のテキスト補正中に予期せぬAPIエラーが発生しました。元のテキストを使用します。エラー: {e}")
        return raw_text, True

# ── マッチング処理 ─────────────────────────────────────────
def match_pages_optimized(
    reader: PdfReader,
    normalized_names_map: Dict[str, str],
    normalized_accounts_map: Dict[str, str],
    pages_to_process: Optional[List[int]] = None,
    use_refine: bool = False,
    status_text_area: Optional[st.empty] = None, # プログレス表示用
    progress_bar_obj: Optional[st.progress] = None, # プログレス表示用
    progress_offset: int = 0, # プログレスバーの開始点
    progress_range: int = 100  # この関数が担当するプログレスの範囲
) -> List[Dict]:
    results = []
    
    if pages_to_process is None:
        page_indices_to_process = range(len(reader.pages))
        total_pages_for_progress = len(reader.pages)
    else:
        page_indices_to_process = [p - 1 for p in pages_to_process if 0 <= p - 1 < len(reader.pages)]
        total_pages_for_progress = len(page_indices_to_process)

    if not page_indices_to_process: # 処理対象ページがない場合は空リストを返す
        return []

    for i, page_idx in enumerate(page_indices_to_process):
        current_page_num = page_idx + 1
        if status_text_area:
            processing_type = "補正マッチング" if use_refine else "基本マッチング"
            status_text_area.text(f"{processing_type}中: ページ {current_page_num} / {len(reader.pages)} (対象リスト内 {i+1}/{total_pages_for_progress})")

        raw_text = reader.pages[page_idx].extract_text() or ""
        if not raw_text.strip(): # 空白のみのテキストもスキップ
            if progress_bar_obj and total_pages_for_progress > 0 :
                progress_bar_obj.progress(progress_offset + int(progress_range * (i + 1) / total_pages_for_progress))
            continue

        text_to_match = raw_text
        api_error_occurred = False

        if use_refine and model:
            # st.spinner を使うと Streamlit のメインの実行フローと干渉することがあるため、
            # ここでは refine_text_with_gemini 内でエラー処理を行う
            text_to_match, api_error_occurred = refine_text_with_gemini(raw_text, current_page_num)
            # APIエラーで元のテキストが返された場合でも、マッチングは試行する

        normalized_page_text = normalize_text(text_to_match) # 空白除去
        
        match_item_found = None

        # 1. 名前でのマッチング
        for norm_name, original_name in normalized_names_map.items():
            if norm_name in normalized_page_text:
                match_item_found = original_name
                break
        
        # 2. 口座番号でのマッチング (名前で見つからなかった場合のみ)
        if not match_item_found:
            page_digits_only = re.sub(r"\D", "", normalized_page_text) # テキストから数字のみ抽出
            if page_digits_only: # 数字が抽出できた場合のみ
                for norm_account, original_account in normalized_accounts_map.items():
                    if norm_account in page_digits_only: # 正規化済みの口座番号と、ページから抽出した数字列を比較
                        match_item_found = original_account
                        break
        
        if match_item_found:
            results.append({
                "page": current_page_num,
                "match": match_item_found,
                "refined_used": use_refine and not api_error_occurred, # 補正が実際に使われたか
                "api_error_on_refine": use_refine and api_error_occurred # 補正時にAPIエラーがあったか
            })
        
        if progress_bar_obj and total_pages_for_progress > 0:
            progress_bar_obj.progress(progress_offset + int(progress_range * (i + 1) / total_pages_for_progress))

    return results


# ── UI フロー ─────────────────────────────────────────
if not pdf_file or not csv_file:
    st.info("サイドバーから PDF と CSV をアップロードしてください。")
    st.stop()

# CSV 読み込み (初回のみ)
@st.cache_data # CSVファイルの内容が変わらない限りキャッシュを利用
def load_and_preprocess_csv(uploaded_file) -> Tuple[pd.DataFrame, Dict[str,str], Dict[str,str]]:
    df = None
    for enc in ("utf-8", "cp932", "shift-jis"): # shift-jisも追加
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, dtype=str, encoding=enc)
            break # 読み込めたらループを抜ける
        except Exception:
            continue
    if df is None:
        st.error("CSV 読み込みに失敗しました。Encoding を確認してください (UTF-8, CP932, Shift-JIS)。")
        st.stop()

    raw_names = df.get("相手方", pd.Series(dtype='str')).dropna().str.strip().tolist()
    # 名前を事前に正規化（空白除去）し、元の名前とマッピング
    # 重複する正規化名がある場合、最初に出現した元の名前を使用
    _normalized_names_map = {normalize_text(name): name for name in reversed(raw_names)} # reversedで処理し、重複時はリスト前方優先

    raw_accounts = []
    for col in ["口座番号１", "口座番号２", "口座番号３"]:
        raw_accounts.extend(df.get(col, pd.Series(dtype='str')).dropna().str.strip().tolist())
    # 口座番号を事前に正規化（数字のみ抽出し、元の口座番号とマッピング）
    _normalized_accounts_map = {re.sub(r"\D", "", acc): acc for acc in reversed(raw_accounts) if re.sub(r"\D", "", acc)}# 空の口座番号を除外

    return df, _normalized_names_map, _normalized_accounts_map

csv_df, normalized_names_map, normalized_accounts_map = load_and_preprocess_csv(csv_file)

st.subheader("CSV プレビュー (先頭5行)")
st.dataframe(csv_df.head())


# PDF bytes 読み込み & PdfReader 初期化 (初回のみ)
@st.cache_data # PDFファイルの内容が変わらない限りキャッシュを利用
def get_pdf_reader(uploaded_file) -> Optional[PdfReader]:
    try:
        uploaded_file.seek(0)
        pdf_bytes_content = uploaded_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes_content))
        if not reader.pages:
            st.error("PDFファイルにページが含まれていません。")
            return None
        return reader
    except Exception as e:
        st.error(f"PDFファイルの読み込みに失敗しました: {e}")
        return None

pdf_reader = get_pdf_reader(pdf_file)

if not pdf_reader:
    st.stop()

st.write(f"PDFファイル: {pdf_file.name}, {len(pdf_reader.pages)} ページ")

# プレビュー実行
if preview_btn:
    st.subheader("プレビュー結果")
    start_preview_time = time.time()
    with st.spinner("プレビュー実行中..."):
        preview_results = match_pages_optimized(
            pdf_reader,
            normalized_names_map,
            normalized_accounts_map,
            use_refine=False # プレビューでは補正なし
        )
    preview_time = time.time() - start_preview_time
    st.success(f"プレビュー完了 ({preview_time:.2f}秒)")
    if preview_results:
        st.table(pd.DataFrame(preview_results).drop(columns=['refined_used', 'api_error_on_refine'], errors='ignore'))
    else:
        st.warning("一致するページが見つかりませんでした。")

# 抽出実行
if extract_btn:
    st.subheader("抽出結果")
    overall_start_time = time.time()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. 基本的なテキストマッチング (全ページ対象)
    status_text.text("ステップ1/4: 基本的なテキストマッチングを実行中...")
    base_matches = match_pages_optimized(
        pdf_reader,
        normalized_names_map,
        normalized_accounts_map,
        use_refine=False,
        status_text_area=status_text,
        progress_bar_obj=progress_bar,
        progress_offset=0,
        progress_range=40 # 全体の40%を割り当て
    )
    progress_bar.progress(40)

    matched_pages_in_base = {m['page'] for m in base_matches}
    
    # 2. 補正マッチング (未マッチページのみ対象)
    pages_for_refinement = [
        p + 1 for p in range(len(pdf_reader.pages)) if (p + 1) not in matched_pages_in_base
    ]
    
    refined_matches = []
    if model and pages_for_refinement:
        status_text.text(f"ステップ2/4: {len(pages_for_refinement)}ページに対して補正マッチングを実行中 (Gemini API)...")
        # st.write(f"補正対象ページ: {pages_for_refinement}") # デバッグ用
        refined_matches = match_pages_optimized(
            pdf_reader,
            normalized_names_map,
            normalized_accounts_map,
            pages_to_process=pages_for_refinement,
            use_refine=True,
            status_text_area=status_text,
            progress_bar_obj=progress_bar,
            progress_offset=40, # 40%からスタート
            progress_range=40   # 全体の40%を割り当て (合計80%)
        )
    elif not model:
        status_text.text("ステップ2/4: Gemini APIが設定されていないため、補正マッチングはスキップされました。")
    else: # pages_for_refinement が空の場合
        status_text.text("ステップ2/4: 全てのページが基本マッチングで処理されたため、補正マッチングはスキップされました。")
    progress_bar.progress(80) # API処理が終わったら80%

    # 3. 結果統合
    status_text.text("ステップ3/4: マッチング結果を統合中...")
    # base_matches を優先し、refined_matches で新たに見つかったものを追加
    final_merged_results = {m['page']: m for m in base_matches}
    for m_refined in refined_matches:
        if m_refined['page'] not in final_merged_results: # refineで新たに見つかったもののみ追加
            final_merged_results[m_refined['page']] = m_refined
    
    # マッチした結果をページ番号順にソート（オプション）
    sorted_matched_items = sorted(final_merged_results.values(), key=lambda x: x['page'])
    progress_bar.progress(85)

    # 4. PDFページ抽出 & ZIP化
    status_text.text(f"ステップ4/4: {len(sorted_matched_items)}ページをZIPファイルにまとめています...")
    zip_buffer = io.BytesIO()
    if sorted_matched_items:
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, item in enumerate(sorted_matched_items):
                    page_no = item['page']
                    match_str = item['match']
                    
                    if status_text: # 詳細な進捗
                        status_text.text(f"ステップ4/4: ページ {page_no} をZIPに追加中... ({i+1}/{len(sorted_matched_items)})")

                    writer = PdfWriter()
                    writer.add_page(pdf_reader.pages[page_no - 1]) # page_no は 1-indexed
                    
                    page_byte_io = io.BytesIO()
                    writer.write(page_byte_io)
                    page_byte_io.seek(0)
                    
                    # ファイル名の禁則文字を置換
                    safe_match_str = re.sub(r'[\\/*?:"<>|]', '_', str(match_str))
                    # ファイル名を短縮（長すぎる場合）
                    safe_match_str = (safe_match_str[:50] + '...') if len(safe_match_str) > 50 else safe_match_str
                    
                    file_name_in_zip = f"{datetime.now():%Y%m%d}_支払通知書_{safe_match_str}_p{page_no}.pdf"
                    zf.writestr(file_name_in_zip, page_byte_io.getvalue())
                    
                    if progress_bar and len(sorted_matched_items) > 0:
                         progress_bar.progress(85 + int(15 * (i + 1) / len(sorted_matched_items))) # 残り15%をZIP化に割り当て
            zip_buffer.seek(0)
        except Exception as e:
            st.error(f"ZIPファイルの作成中にエラーが発生しました: {e}")
            zip_buffer = None # エラー時はNoneにする
    
    progress_bar.progress(100)
    overall_time = time.time() - overall_start_time
    
    if sorted_matched_items and zip_buffer:
        st.success(f"抽出完了 ({overall_time:.2f}秒) - {len(sorted_matched_items)}ページをZIP化しました。")
        st.download_button(
            label="ZIPダウンロード",
            data=zip_buffer,
            file_name=f"{datetime.now():%Y%m%d}_支払通知書_{pdf_file.name if pdf_file else ''}.zip",
            mime="application/zip"
        )
        # 抽出結果の詳細テーブル表示（オプション）
        st.write("抽出されたページとマッチング情報:")
        df_results = pd.DataFrame(sorted_matched_items)
        st.dataframe(df_results)

    elif not sorted_matched_items:
        st.warning(f"抽出対象のページが見つかりませんでした。({overall_time:.2f}秒)")
    else: # zip_buffer が None の場合 (エラー発生時)
        st.error(f"処理は完了しましたが、ZIPファイルの準備に失敗しました。({overall_time:.2f}秒)")

    status_text.text(f"処理が完了しました。 ({overall_time:.2f}秒)")
