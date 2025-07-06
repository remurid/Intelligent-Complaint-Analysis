import unittest
from src.core.vector_store_manager import VectorStoreManager

class TestVectorStoreManager(unittest.TestCase):
    def test_add_and_query(self):
        db_path = 'test_db'
        collection_name = 'test_collection'
        manager = VectorStoreManager(db_path, collection_name)
        docs = ["test document"]
        metas = [{"id": "1"}]
        ids = ["1"]
        manager.add_documents(docs, metas, ids)
        result = manager.query(["test"], n_results=1)
        self.assertTrue('documents' in result)

if __name__ == "__main__":
    unittest.main()
