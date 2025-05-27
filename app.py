import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import io

# 必要に応じて pytesseract のインポートを囲みます（環境によってはインポートできないため）
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    st.error('pytesseract がインストールされていません。OCR 機能は利用できません。')

def read_pdf_text(pdf_file):
    """
    PDFを1ページずつ画像としてOCRで読み取り、各ページのテキストをリストで返す。
    """
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    texts = []
    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if TESSERACT_AVAILABLE:
            text = pytesseract.image_to_string(img, lang='jpn')
        else:
            text = '(OCR不可)'
        texts.append(text)
    return texts

def load_csv(csv_file):
    # 自動判別でエンコーディングを推測
    try:
        df = pd.read_csv(csv_file, dtype=str, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, dtype=str, encoding='cp932')
        except Exception as e:
            st.error(f'CSV読み込みエラー: {e}')
            return None
    return df

def main():
    st.title('PDF名寄せOCR照合アプリ')
    st.write('PDFに書かれている名前（OCRで抽出）とCSVの名簿を照合します。')

    pdf_file = st.file_uploader('PDFをアップロード', type=['pdf'])
    csv_file = st.file_uploader('CSV名簿をアップロード', type=['csv'])

    if pdf_file and csv_file:
        # CSV読み込み
        csv_df = load_csv(csv_file)
        if csv_df is None:
            st.stop()
        # OCRでPDFテキスト抽出
        st.info('PDFをOCRで読み取り中...')
        pdf_texts = read_pdf_text(pdf_file)

        # 名前候補抽出（CSVの1列目 or "名前"列優先）
        name_col = None
        for c in csv_df.columns:
            if '名' in c:
                name_col = c
                break
        if name_col is None:
            name_col = csv_df.columns[0]  # 1列目をデフォルト
        names = csv_df[name_col].fillna('').tolist()

        # 照合結果
        result = []
        for page_num, text in enumerate(pdf_texts):
            found = []
            for n in names:
                # 名前がOCR結果に現れるかを判定（完全一致 or 部分一致）
                if n and (n in text):
                    found.append(n)
            result.append({
                'page': page_num + 1,
                'names_found': ', '.join(found) if found else '(該当なし)'
            })

        st.success('照合結果')
        st.dataframe(pd.DataFrame(result))
        # 詳細表示
        with st.expander('PDF各ページのOCRテキスト詳細'):
            for i, txt in enumerate(pdf_texts):
                st.markdown(f'**ページ{i+1}**\n{text}')

if __name__ == '__main__':
    main()
