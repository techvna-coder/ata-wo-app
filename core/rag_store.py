# core/rag_store.py
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_faiss(vector_dir: str, openai_api_key: str):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    store = FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)
    return store

def similarity_search_with_filter(store, query: str, k=4, doc_type: str = None):
    if doc_type:
        return store.similarity_search_with_score(query, k=k, filter={"doc_type": doc_type})
    return store.similarity_search_with_score(query, k=k)
