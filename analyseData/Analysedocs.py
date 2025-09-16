from docx import Document  # 处理Word文档

def load_docx(file_path):
    """加载Word文档内容"""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)
