"""
ComplaintProcessor: Handles loading and cleaning of complaint data.
"""
from typing import Any
import pandas as pd

class ComplaintProcessor:
    def __init__(self, config: Any):
        self.config = config

    def load_data(self, path: str) -> pd.DataFrame:
        """Load complaint data from a CSV file."""
        return pd.read_csv(path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the complaint data (lowercase, remove special chars, etc)."""
        df = df.copy()
        df['Consumer complaint narrative'] = df['Consumer complaint narrative'].astype(str)
        df = df[df['Consumer complaint narrative'].str.strip() != '']
        df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.lower()
        # Add more cleaning steps as needed
        return df
