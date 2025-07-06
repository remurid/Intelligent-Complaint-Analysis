import unittest
from src.core.embedder import Embedder

class TestEmbedder(unittest.TestCase):
    def test_embed(self):
        model_name = 'all-MiniLM-L6-v2'
        embedder = Embedder(model_name)
        texts = ["hello world", "test embedding"]
        embeddings = embedder.embed(texts)
        self.assertEqual(len(embeddings), 2)

if __name__ == "__main__":
    unittest.main()
