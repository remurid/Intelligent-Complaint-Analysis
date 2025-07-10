import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import time
import os
import shutil

# --- Configuration ---
INPUT_CSV_PATH = './data/filtered_complaints.csv'
DB_PATH = 'complaint_db'
COLLECTION_NAME = 'financial_complaints'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
REBUILD_DB = True  # Set to True to delete and recreate the collection

def main():
    print("\n--- Starting Task 2: Vector Store Creation ---")

    # 1. Load cleaned data
    print(f"\n[1/5] Loading cleaned data from '{INPUT_CSV_PATH}'...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print("Available columns:", df.columns.tolist())
        print(f"-> Loaded {len(df)} complaints.")
    except FileNotFoundError:
        print("❌ Error: Cleaned CSV not found. Run Task 1 first.")
        return

    # 2. Chunk narratives
    print(f"\n[2/5] Chunking narratives (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    df['chunks'] = df['Consumer complaint narrative'].astype(str).apply(lambda x: splitter.split_text(x))
    print("-> Chunking complete.")

    # 3. Setup ChromaDB
    if REBUILD_DB and os.path.exists(DB_PATH):
        print(f"\n[3/5] Rebuilding ChromaDB. Deleting old DB at '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)

    print("-> Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"-> Collection ready: '{COLLECTION_NAME}'")

    # 4. Embed and Add Documents
    print("\n[4/5] Embedding and storing chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    doc_count = 0

    start_time = time.time()

    for idx, row in df.iterrows():
        product = row['Product']
        chunks = row['chunks']
        base_id = f"row{idx}"

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{base_id}_{i}"
            vector = model.encode(chunk_text)

            collection.add(
                documents=[chunk_text],
                embeddings=[vector.tolist()],
                metadatas=[{
                    'product': product,
                    'source_row': idx
                }],
                ids=[chunk_id]
            )
            doc_count += 1

    end_time = time.time()
    print(f"-> Added {doc_count} chunks to vector store.")
    print(f"-> Total documents in collection now: {collection.count()}")
    print(f"-> Time taken: {end_time - start_time:.2f}s")

    # 5. Verify
    print("\n[5/5] Running verification query...")
    try:
        query_text = "My loan was wrongly charged"
        results = collection.query(query_texts=[query_text], n_results=3)

        print(f"\n--- Results for query: '{query_text}' ---")
        if not results['documents'][0]:
            print("⚠️  No results found. Check if chunking or filtering was too strict.")
        else:
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nResult {i+1}:")
                print(f"  Text: {doc[:150]}...")
                print(f"  Metadata: {results['metadatas'][0][i]}")
                print(f"  Distance: {results['distances'][0][i]:.4f}")
    except Exception as e:
        print(f"❌ Verification failed: {e}")

    print("\n✅ Task 2 Complete. Vector DB is ready in:", DB_PATH)

if __name__ == '__main__':
    main()