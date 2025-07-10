# 💬 Intelligent Complaint Analysis: RAG Chatbot

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot for analyzing and answering financial complaints using real-world data from the **Consumer Financial Protection Bureau (CFPB)**.  

It uses **semantic search** via ChromaDB and **large language models (LLMs)** to return insightful, grounded answers backed by actual complaint narratives.

---

## 📦 Installation

```bash
git clone https://github.com/remurid/Intelligent-Complaint-Analysis.git
cd Intelligent-Complaint-Analysis
pip install -r requirements.txt
```

> 🧪 Python 3.10+ is recommended  
> 🧠 Make sure you have `transformers`, `chromadb`, `sentence-transformers`, and `streamlit`.

---

## 🚀 Project Workflow

### ✅ Step 1: Data Preprocessing & EDA
Explore and clean raw complaints in the notebook:

```bash
notebooks/eda_preprocessing.ipynb
```

This will output a cleaned dataset to:

```
data/filtered_complaints.csv
```

---

### ✅ Step 2: Chunking, Embedding, and Vector Store Indexing

Generate vector embeddings and build the ChromaDB vector store:

```bash
python scripts/chunk_embed_index.py
```

Or use the modular script:

```bash
python scripts/task2_create_vector_store.py
```

→ Output is stored in:  
```
vector_store_chroma/chroma_db/
```

---

### ✅ Step 3: Build and Evaluate the RAG Pipeline

Run qualitative evaluation using:

```bash
python scripts/rag_pipeline.py
```

This retrieves relevant complaint chunks, sends them to an LLM, and prints the AI-generated answers.

---

### ✅ Step 4: Interactive Chat Interface (Streamlit)

Launch the chatbot UI:

```bash
streamlit run app.py
```

Features:
- Input box for questions
- Answer generated using retrieved chunks
- Retrieved source texts shown for transparency
- “Clear chat” button

---

## 🗂 Project Structure

```
├── app.py                         # Streamlit chatbot interface
├── scripts/
│   ├── chunk_embed_index.py      # Chunking + embedding script (Task 2)
│   ├── task2_create_vector_store.py # Alt Task 2 version (modular)
│   └── rag_pipeline.py           # RAG logic + evaluation (Task 3)
├── src/
│   ├── core/                     # Reusable class-based modules (optional)
│   └── utils/                    # Helper functions or configs
├── data/
│   └── filtered_complaints.csv   # Cleaned data (from Task 1)
├── notebooks/
│   └── eda_preprocessing.ipynb   # Task 1: EDA + preprocessing
├── vector_store_chroma/
│   ├── chroma_db/                # Persisted ChromaDB vector store
│   └── metadata.pkl              # Optional metadata backup
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── tests/                        # (Optional) unit tests
```

---

## 🧠 Tech Stack

| Component        | Tool                         |
|------------------|------------------------------|
| Language Model   | `google/flan-t5-base`        |
| Embedding Model  | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database  | `ChromaDB`                   |
| Frontend UI      | `Streamlit`                  |
| Chunking         | `LangChain` TextSplitter     |
| Dataset          | CFPB Complaint Dataset       |

---

## 📝 Notes

- Retrieval is semantic — results are based on meaning, not keywords.
- The chatbot is transparent: it always shows the source texts used in the answer.
- Modular architecture makes it easy to plug in other models, databases, or UIs.


---

## 🤝 Acknowledgements

This project was developed as part of the **10 Academy AI Mastery Program – Week 6 Challenge**.

---

## 📬 Contact

> Remedan Ridwan – [rexrid1@gmail.com]  
> GitHub: [github.com/remurid]