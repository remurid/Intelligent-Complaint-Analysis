import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import time

# --- Configuration ---
INPUT_CSV_PATH = './data/filtered_complaints.csv'  # Path to the cleaned data from Task 1
DB_PATH = "complaint_db"                    # Directory to save the vector database
COLLECTION_NAME = "financial_complaints"    # Name of the collection in ChromaDB
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'   # Hugging Face model for embeddings
CHUNK_SIZE = 512                            # Max characters per text chunk
CHUNK_OVERLAP = 50                          # Overlap between consecutive chunks

def main():
    """
    Main function to execute the entire data processing and indexing pipeline.
    """
    print("--- Starting Task 2: Vector Store Creation ---")

    # 1. Load Cleaned Data
    # ---------------------
    try:
        print(f"\n[1/5] Loading cleaned data from '{INPUT_CSV_PATH}'...")
        df = pd.read_csv(INPUT_CSV_PATH)
        # Ensure the narrative column is treated as a string and drop any empty rows
        df['Consumer complaint narrative'] = df['Consumer complaint narrative'].astype(str)
        df.dropna(subset=['Consumer complaint narrative'], inplace=True)
        print(f"-> Successfully loaded {len(df)} complaints.")
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
        print("Please make sure you have run Task 1 and the output CSV is in the correct location.")
        return

    # 2. Chunk the Text Narratives
    # ----------------------------
    print(f"\n[2/5] Chunking text narratives (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    # The .apply method runs the splitter on each narrative in the DataFrame
    df['chunks'] = df['Consumer complaint narrative'].apply(lambda x: text_splitter.split_text(x))
    print("-> Text chunking complete.")

    # 3. Initialize Embedding Model and Vector Store
    # ----------------------------------------------
    print(f"\n[3/5] Initializing embedding model ('{EMBEDDING_MODEL_NAME}') and ChromaDB...")
    
    # Initialize ChromaDB client for persistent storage
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create or get the collection. This is where the vectors will be stored.
    # We pass the embedding model name to the collection metadata.
    # ChromaDB can use this to automatically handle embeddings if needed.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Specifies the distance metric for similarity search
    )
    print(f"-> ChromaDB client initialized. Using collection: '{COLLECTION_NAME}'.")

    # 4. Process, Embed, and Store Documents
    # ----------------------------------------
    print("\n[4/5] Processing and storing documents in the vector store...")
    start_time = time.time()
    
    # Keep track of documents already in the collection to avoid duplicates
    existing_ids = set(collection.get(include=[])['ids'])
    print(f"-> Found {len(existing_ids)} existing documents in the collection.")
    
    doc_id_counter = 0
    new_docs_added = 0

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        complaint_id = str(row['Complaint ID'])
        product = row['Product']
        
        # Loop through each chunk of the current complaint
        for i, chunk_text in enumerate(row['chunks']):
            # Create a unique ID for each chunk to prevent duplicates
            doc_id = f"{complaint_id}_{i}"
            
            if doc_id not in existing_ids:
                # Add the document to the collection
                collection.add(
                    documents=[chunk_text],
                    metadatas=[{
                        'product': product,
                        'complaint_id': complaint_id 
                    }],
                    ids=[doc_id]
                )
                new_docs_added += 1

    end_time = time.time()
    print(f"-> Processing complete.")
    print(f"-> Added {new_docs_added} new documents to the collection.")
    print(f"-> Total documents in collection: {collection.count()}.")
    print(f"-> Time taken: {end_time - start_time:.2f} seconds.")

    # 5. Verify the Vector Store
    # --------------------------
    print("\n[5/5] Running a verification query...")
    try:
        query_text = "My bank charged me an incorrect overdraft fee"
        results = collection.query(
            query_texts=[query_text],
            n_results=3  # Find the top 3 most similar chunks
        )

        print(f"\n--- Verification Results for Query: '{query_text}' ---")
        if not results['documents'][0]:
             print("-> Query returned no results. The database might be empty or the query too specific.")
        else:
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nResult {i+1}:")
                print(f"  Text: '{doc[:150].strip()}...'") # Print the first 150 characters
                print(f"  Metadata: {results['metadatas'][0][i]}")
                print(f"  Distance: {results['distances'][0][i]:.4f}")
    except Exception as e:
        print(f"An error occurred during verification: {e}")

    print("\n--- Task 2 Complete ---")
    print(f"Your vector store is ready and saved in the '{DB_PATH}' directory.")

if __name__ == '__main__':
    main()
