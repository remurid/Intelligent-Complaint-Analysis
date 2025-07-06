# Intelligent Complaint Analysis: RAG Chatbot

This project builds a Retrieval-Augmented Generation (RAG) chatbot for answering financial complaints using the CFPB dataset. It features modular, object-oriented code for data cleaning, chunking, embedding, and vector store indexing for efficient semantic search and retrieval.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing & EDA
Run the notebook in `notebooks/eda_preprocessing.ipynb` to explore and clean the complaint data. The cleaned data will be saved to `data/filtered_complaints.csv`.

### 2. Chunking, Embedding, and Indexing
Run the following script to chunk narratives, generate embeddings, and build a vector store:

```bash
python scripts/chunk_embed_index.py
```
This will create a persistent ChromaDB vector store in the `complaint_db/` directory.

## Project Structure

- `src/core/`: Modular, object-oriented code for data processing, chunking, embedding, and vector storage
- `src/utils/`: Configuration and utility modules
- `notebooks/eda_preprocessing.ipynb`: EDA and data cleaning notebook
- `scripts/chunk_embed_index.py`: Script for chunking, embedding, and vector store creation
- `data/filtered_complaints.csv`: Cleaned and filtered complaint data
- `complaint_db/`: Directory containing the ChromaDB vector store
- `tests/unit/`: Unit tests for core modules
- `docs/`: Documentation

## Requirements
See `requirements.txt` for all dependencies, including pandas, langchain, sentence-transformers, and chromadb.

## Notes
- The embedding model used is `all-MiniLM-L6-v2` (sentence-transformers).
- ChromaDB is used for fast vector search and retrieval.
- The codebase is modular and object-oriented for maintainability and extensibility.
