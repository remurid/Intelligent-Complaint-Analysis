import unittest
from src.core.text_chunker import TextChunker

class TestTextChunker(unittest.TestCase):
    def test_chunk(self):
        chunker = TextChunker(chunk_size=5, overlap=2)
        text = "abcdefghij"
        chunks = chunker.chunk(text)
        self.assertEqual(chunks, ["abcde", "defgh", "ghij"])

if __name__ == "__main__":
    unittest.main()
