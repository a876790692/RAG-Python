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
rom langchain.vectorstores import FAISS
import os
import pickle


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


def load_or_create_vector_db(documents_dir, index_path="analyseData/faiss", embedding_model_path=r"C:\Users\325043\Desktop\model\all-MiniLM-L6-v2"):
    """加载或创建 FAISS 向量数据库"""
    
    # 如果存在已保存的索引，直接加载
    if os.path.exists(index_path):
        print("✅ 加载已保存的向量数据库...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
        return FAISS.load_local(index_path, embeddings)
    
    # 否则，创建新的向量库
    print("🔄 未找到向量数据库，开始重新构建...")
    all_texts = []
    for file_name in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
            ext = file_name.split('.')[-1].lower()
            text = load_file(file_path, ext)
            all_texts.append({'content': text, 'filename': file_name})

    if not all_texts:
        raise ValueError("❌ 没有找到任何文档，请检查目录路径")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    metadatas = [{"source": doc['filename']} for doc in all_texts]

    db = FAISS.from_texts([doc['content'] for doc in all_texts], embeddings, metadatas=metadatas)
    db.save_local(index_path)
    print("✅ 向量数据库已创建并保存")

    return db

# ===============================
# 初始化本地 Hugging Face LLM
# ===============================
def init_llm(model_name_or_path=r"C:\Users\325043\Desktop\model\DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    初始化本地 Hugging Face 模型
    model_name_or_path: Hugging Face Hub 名称或本地路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True, cache_dir='/home/{username}/huggingface')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=512,
        do_sample=False
    )

    # 用 langchain 的 HuggingFacePipeline 封装
    return HuggingFacePipeline(pipeline=pipe)

def rag_pipeline(documents_dir, user_query, model_name=r"C:\Users\325043\Desktop\model\DeepSeek-R1-Distill-Qwen-1.5B"):
    """RAG 流程：加载或使用已保存的向量索引 → 问答"""
    
    # 1. 加载或创建向量数据库
    db = load_or_create_vector_db(documents_dir)

    # 2. 初始化本地 Hugging Face 模型
    llm = init_llm(model_name_or_path=model_name)

    # 3. 创建检索器
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    # 4. 创建问答链
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
    documents_dir = "analyseData/lib"
    #documents_dir = r""
    query = input("请输入你的问题: ")

    # 使用 Hugging Face 本地模型
    result = rag_pipeline(documents_dir, query, model_name="analyseData/modle/DeepSeek-R1-Distill-Qwen-1.5B")

    if isinstance(result, dict):
        print("\n📌 答案:")
        print(result["answer"])

        print("\n📌 相关内容来源（片段）:")
        for src in result["sources"]:
            print(f"- 来源: {src['filename']}")
            print(f"  内容: {src['text']}\n")
    else:
        print(result)
