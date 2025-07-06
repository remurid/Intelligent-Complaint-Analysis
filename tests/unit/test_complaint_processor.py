import unittest
import pandas as pd
from src.core.complaint_processor import ComplaintProcessor

class DummyConfig:
    pass

class TestComplaintProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ComplaintProcessor(DummyConfig())
        self.df = pd.DataFrame({
            'Consumer complaint narrative': ['Test Complaint!', '', 'Another complaint.']
        })

    def test_clean_data(self):
        cleaned = self.processor.clean_data(self.df)
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(all(cleaned['Consumer complaint narrative'].str.islower()))

if __name__ == "__main__":
    unittest.main()
