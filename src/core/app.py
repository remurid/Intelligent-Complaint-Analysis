import streamlit as st
from scripts.rag_pipeline import RAGPipeline

# --- Configuration ---
DB_PATH = "./complaint_db"
COLLECTION_NAME = "financial_complaints"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# --- Initialize RAG ---
@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL
    )

rag = load_pipeline()

# --- UI Layout ---
st.set_page_config(page_title="ComplaintBot", layout="wide")
st.title("🧠 Complaint Insight Chatbot")
st.caption("Ask me questions about real financial service complaints. I'll answer using actual complaint narratives.")

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input Area ---
question = st.text_input("💬 Ask a question:", placeholder="e.g., Why are people unhappy with BNPL?")

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("🔍 Ask")
with col2:
    clear = st.button("♻️ Clear Chat")

# --- Handle "Clear" Button ---
if clear:
    st.session_state.history = []
    st.experimental_rerun()

# --- Handle Question Submission ---
if ask and question.strip() != "":
    with st.spinner("Thinking..."):
        answer, sources = rag.answer_question(question)

    # Save to history
    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })

# --- Display Chat History ---
for item in reversed(st.session_state.history):
    st.markdown(f"**🧑 You:** {item['question']}")
    st.markdown(f"**🤖 ComplaintBot:** {item['answer']}")

    with st.expander("📄 Show Retrieved Sources"):
        for i, src in enumerate(item['sources']):
            st.markdown(f"**Source {i+1}:**")
            st.code(src)

    st.divider()
