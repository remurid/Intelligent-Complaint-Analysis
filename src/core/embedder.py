"""
Embedder: Handles embedding of text chunks using sentence-transformers.
"""
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]):
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
