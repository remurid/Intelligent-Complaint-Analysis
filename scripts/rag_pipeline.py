import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import textwrap

# --- Configuration ---
DB_PATH = "./complaint_db"                  # Path to the vector database created in Task 2
COLLECTION_NAME = "financial_complaints"    # Collection name in ChromaDB
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'   # Embedding model for queries
LLM_MODEL_NAME = 'google/flan-t5-base'      # LLM for text generation (using 'base' for lower resource usage)
TOP_K_RESULTS = 3                           # Number of relevant chunks to retrieve

class RAGPipeline:
    """
    A class to encapsulate the entire RAG pipeline from retrieval to generation.
    """
    def __init__(self, db_path, collection_name, embedding_model_name, llm_model_name):
        """
        Initializes the RAG pipeline components.
        """
        print("Initializing RAG pipeline...")
        
        # 1. Initialize Retriever (ChromaDB client and embedding model)
        print(" -> Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        print(f" -> Connecting to vector store at: {db_path}")
        try:
            db_client = chromadb.PersistentClient(path=db_path)
            self.collection = db_client.get_collection(name=collection_name)
            
            # --- DIAGNOSTIC STEP ---
            # Check if the collection is empty. This is a crucial check.
            collection_count = self.collection.count()
            print(f" -> Collection '{collection_name}' loaded successfully.")
            print(f" -> The collection contains {collection_count} documents.")
            if collection_count == 0:
                print(" -> WARNING: The collection is empty. The retriever will not find any documents.")
                print(" -> Please ensure that Task 2 (create_vector_store.py) ran correctly and populated the database.")

        except Exception as e:
            print(f"FATAL ERROR: Could not connect to the vector store at '{db_path}'.")
            print(f"Please ensure the path is correct and the database was created in Task 2.")
            print(f"Error details: {e}")
            raise

        # 2. Initialize Generator (LLM pipeline)
        print(" -> Loading Large Language Model...")
        self.generator = pipeline(
            'text2text-generation',
            model=llm_model_name,
            max_length=512  # Adjust max_length as needed for your answers
        )
        print("RAG pipeline initialized successfully.")

    def _retrieve_context(self, question, n_results=TOP_K_RESULTS):
        """
        Retrieves relevant context from the vector store by first embedding the user's question.
        """
        # 1. Embed the user's question using the same model used for the documents.
        #    This makes the search process explicit and reliable.
        query_embedding = self.embedding_model.encode(question)
        
        # 2. Query the collection using the generated embedding vector.
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], # Pass the embedding vector
            n_results=n_results
        )
        return results['documents'][0]

    def _create_prompt(self, question, context):
        """
        Creates a detailed prompt for the LLM, including the retrieved context.
        """
        context_str = "\n\n".join(context)
        
        # This prompt template is critical for guiding the LLM's behavior.
        prompt_template = f"""
You are a helpful financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based ONLY on the provided context.

Follow these rules:
1.  Synthesize an answer directly from the information given in the "CONTEXT" section.
2.  Do not use any outside knowledge or make up information.
3.  If the context does not contain the answer to the question, you MUST state: "I do not have enough information in the provided context to answer this question."
4.  Quote or reference specific parts of the context to support your answer where possible.

CONTEXT:
---
{context_str}
---

QUESTION:
{question}

ANSWER:
"""
        return prompt_template

    def answer_question(self, question):
        """
        The main method to answer a question using the RAG pipeline.
        """
        # 1. Retrieve context
        retrieved_context = self._retrieve_context(question)
        
        # 2. Create the prompt
        prompt = self._create_prompt(question, retrieved_context)
        
        # 3. Generate the answer
        print("\nGenerating answer...")
        generated_text = self.generator(prompt)
        answer = generated_text[0]['generated_text']
        
        return answer, retrieved_context


def run_evaluation(pipeline_instance, questions):
    """
    Runs a qualitative evaluation on a list of questions and prints the results.
    """
    print("\n--- Starting Qualitative Evaluation ---")
    wrapper = textwrap.TextWrapper(width=100) # For pretty printing

    for i, question in enumerate(questions):
        print(f"\n==================== Question {i+1} ====================")
        print(f"Question: {question}")
        
        answer, sources = pipeline_instance.answer_question(question)
        
        print("\n--- Generated Answer ---")
        print(wrapper.fill(answer))
        
        print("\n--- Retrieved Sources (Context) ---")
        if not sources:
            print("  -> No sources were retrieved from the database.")
        else:
            for j, source in enumerate(sources):
                print(f"Source {j+1}:")
                print(wrapper.fill(f"  -> {source}"))
        
        print("\n--- Evaluation ---")
        print("Quality Score (1-5): [Your Score Here]")
        print("Comments/Analysis:   [Your Analysis Here]")
        print("====================================================\n")


if __name__ == '__main__':
    try:
        # Initialize the RAG pipeline
        rag_pipe = RAGPipeline(
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            llm_model_name=LLM_MODEL_NAME
        )

        # Define the list of evaluation questions as required by the assignment
        evaluation_questions = [
            "Why are people unhappy with the Buy Now, Pay Later (BNPL) service?",
            "What are the most common complaints about credit card billing disputes?",
            "Are there any recurring issues with money transfers being delayed?",
            "What problems are customers facing when trying to close their savings accounts?",
            "Describe a situation where a customer was wrongly charged for a personal loan."
        ]

        # Run the evaluation
        run_evaluation(rag_pipe, evaluation_questions)
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
