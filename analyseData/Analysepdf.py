from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader  # 处理PDF文件

def load_pdf(file_path):
    """加载PDF文件内容"""
    text = ""
    with open(file_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # 处理可能的None值
    return text