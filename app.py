# --- Streamlit 完全アプリ例：PDF画像＋CSV照合 ---
# 画像上のマーカー名をOCRし、CSVとマッチしてハイライト表示
# 必要パッケージ: streamlit, pandas, pillow, pytesseract
# 注意: pytesseract実行環境/日本語OCRはtesseract本体+言語データ必須

import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import io

# --- CSV読込：複数エンコード自動判定 ---
def read_csv_any_encoding(uploaded_file):
    encodings = ['utf-8', 'shift_jis', 'cp932', 'utf-16']
    for enc in encodings:
        uploaded_file.seek(0)  # ファイルポインタを先頭に戻す
        try:
            df = pd.read_csv(uploaded_file, encoding=enc, dtype=str)
            return df
        except Exception as e:
            continue
    st.error('CSVファイルの読み込みに失敗しました。エンコードを確認してください。')
    return None

# --- 画像からマーカ部分のOCR抽出 ---
def ocr_image(image, lang='jpn'):
    # 必要に応じて領域指定 or 前処理も追加
    text = pytesseract.image_to_string(image, lang=lang)
    return text

# --- メイン画面 ---
st.title('PDF画像OCR×CSVマッチングツール')

uploaded_img = st.file_uploader('画像ファイルをアップロード (PNG/JPG)', type=['png','jpg','jpeg'])
uploaded_csv = st.file_uploader('CSVファイルをアップロード', type=['csv'])

if uploaded_img and uploaded_csv:
    image = Image.open(uploaded_img)
    st.image(image, caption='アップロード画像', use_column_width=True)
    
    # OCR結果表示
    st.subheader('OCR抽出結果（画像全体）')
    ocr_txt = ocr_image(image, lang='jpn')
    st.code(ocr_txt)

    # CSV読込
    df = read_csv_any_encoding(uploaded_csv)
    if df is not None:
        st.subheader('CSVプレビュー')
        st.dataframe(df)
        
        # CSV内のどの列と照合するか選択
        colname = st.selectbox('照合に使うCSV列名を選択してください', df.columns)

        # OCR抽出テキストとマッチする行を抽出
        matches = df[df[colname].astype(str).apply(lambda x: any(word in ocr_txt for word in x.split()))]

        st.subheader('マッチした行')
        if not matches.empty:
            st.dataframe(matches)
        else:
            st.info('OCR結果とマッチする行が見つかりませんでした。')
else:
    st.info('画像ファイルとCSVファイルの両方をアップロードしてください。')

# --- 補足説明・イメージ図 ---
st.markdown("""
---
### 【イメージ図】

以下の流れで処理を行います：

1. ![Step1](https://img.icons8.com/ios-filled/50/000000/upload.png) 画像ファイルアップロード → 画像内のオレンジマーカー部分をOCRでテキスト化
2. ![Step2](https://img.icons8.com/ios-filled/50/000000/spreadsheet.png) CSVファイルアップロード → 名前などと照合
3. ![Step3](https://img.icons8.com/ios-filled/50/000000/search.png) OCRテキストとCSVをマッチング → 一致行のみ表示

""")
