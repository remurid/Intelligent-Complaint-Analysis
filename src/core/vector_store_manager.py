"""
VectorStoreManager: Handles storing and querying embeddings in ChromaDB.
"""
from typing import List, Dict, Any
import chromadb

class VectorStoreManager:
    def __init__(self, db_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_texts: List[str], n_results: int = 3):
        return self.collection.query(query_texts=query_texts, n_results=n_results)
