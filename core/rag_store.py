# core/rag_store.py

# Fallback cho FAISS (bản mới/cũ)
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # fallback cho bản cũ

# Fallback cho OpenAIEmbeddings
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    from langchain.embeddings import OpenAIEmbeddings  # fallback rất cũ

def load_faiss(vector_dir: str, openai_api_key: str):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    store = FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)
    return store

def similarity_search_with_filter(store, query: str, k=4, doc_type: str = None):
    # Một số backend không hỗ trợ filter metadata trực tiếp; có thể lọc thủ công sau khi lấy top-k lớn hơn.
    return store.similarity_search_with_score(query, k=k)
