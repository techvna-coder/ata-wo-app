"""
RAG store (stub) – tránh lỗi import khi môi trường chưa cài langchain/faiss.
Cung cấp API tối thiểu tương thích: load_faiss() trả về None.
"""
from typing import Optional, Any

def load_faiss(index_dir: str) -> Optional[Any]:
    """
    Trả về None để biểu thị chưa bật RAG/FAISS.
    Triển khai thật sẽ trả về đối tượng vector store và phương thức search.
    """
    return None
