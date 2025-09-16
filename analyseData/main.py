# main.py
import os
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import Analysepdf as Apdf
import Analysedocs as Adoc

# ===============================
# 文件加载函数
# ===============================
def load_file(file_path, file_type):
    """根据文件类型加载不同文档"""
    if file_type == 'pdf':
        return Apdf.load_pdf(file_path)
    elif file_type == 'docx':
        return Adoc.load_docx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

# ===============================
# 初始化本地 Ollama LLM
# ===============================
def init_llm(model_name="deepseek-r1:8b"):
    """初始化本地 Ollama 模型"""
    return OllamaLLM(
        model=model_name,
        temperature=0
    )
# ===============================
# RAG 主流程
# ===============================
def rag_pipeline(documents_dir, user_query, model_name="deepseek-r1:8b"):
    """RAG 流程：加载文档 → 向量索引 → 问答"""
    # 1. 加载文档
    all_texts = []
    for file_name in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
            ext = file_name.split('.')[-1].lower()
            text = load_file(file_path, ext)
            all_texts.append({'content': text, 'filename': file_name})

    if not all_texts:
        return "❌ 没有找到任何文档，请检查目录路径"

    print("✅ 文档已加载，创建向量索引...")

    # 2. 嵌入模型（这里用 HuggingFace sentence-transformer，本地可跑）
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. 创建向量数据库
    metadatas = [{"source": doc['filename']} for doc in all_texts]
    db = FAISS.from_texts(
        [doc['content'] for doc in all_texts],
        embeddings,
        metadatas=metadatas
    )
    db.save_local("faiss_index")
    print("✅ 向量数据库已创建")

    # 4. 初始化本地 LLM
    llm = init_llm(model_name=model_name)

    # 5. 创建检索器
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    # 6. 创建问答链（新版写法）
    prompt_template = "根据以下文档回答问题：\n{context}\n\n问题: {question}\n答案:"
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    print("✅ 问答链已准备好，正在回答问题...")

    try:
        response = qa_chain.invoke({"query": user_query})

        # 提取答案和来源
        answer = response["result"]
        source_docs = response.get("source_documents", [])

        sources = []
        for doc in source_docs:
            metadata = doc.metadata
            page_content = doc.page_content
            if isinstance(metadata, dict) and "source" in metadata:
                snippet = page_content[:50] + "..." if len(page_content) > 50 else page_content
                sources.append({"filename": metadata["source"], "text": snippet})

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"❌ 处理查询时出错: {e}")
        return "抱歉，我无法回答这个问题，请提供更多上下文信息。"


# ===============================
# 主程序入口
# ===============================
if __name__ == "__main__":
    documents_dir = "./my_documents"
    query = input("请输入你的问题: ")

    # 默认使用本地 deepseek-r1:8b 模型
    result = rag_pipeline(documents_dir, query, model_name="deepseek-r1:8b")

    if isinstance(result, dict):
        print("\n📌 答案:")
        print(result["answer"])

        print("\n📌 相关内容来源（片段）:")
        for src in result["sources"]:
            print(f"- 来源: {src['filename']}")
            print(f"  内容: {src['text']}\n")
    else:
        print(result)
