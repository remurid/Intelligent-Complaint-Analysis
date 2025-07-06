"""
Project configuration settings.
"""

class Config:
    DATA_PATH = './data/filtered_complaints.csv'
    DB_PATH = 'complaint_db'
    COLLECTION_NAME = 'financial_complaints'
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
