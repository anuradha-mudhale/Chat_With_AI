import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# --- Load env ---
load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="PDF Chat",
    page_icon="💬",
    layout="wide"
)

# --- LLM ---
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

llm = get_llm()

# --- Session ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --- CSS (FIXED UI) ---
st.markdown("""
<style>
html, body, .stApp {
    background-color: ##ffffff !important;
    color: #ececec !important;
}

/* Hide default */
#MainMenu, footer, header {visibility:hidden}

/* USER MESSAGE */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) 
div[data-testid="stMarkdownContainer"] {
    background: #10a37f !important;
    color: white !important;
    padding: 12px;
    border-radius: 12px;
}

/* ASSISTANT MESSAGE */
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) 
div[data-testid="stMarkdownContainer"] {
    background: #2a2a2a !important;
    color: #f1f1f1 !important;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #3a3a3a;
}

/* Chat input */
[data-testid="stChatInput"] > div {
    background: #3a3a3a !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# SIDEBAR
# ─────────────────────────────
with st.sidebar:
    st.title("💬 Chat With AI")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(documents)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.from_documents(docs, embeddings)
            retriever = db.as_retriever()

            # Prompt
            prompt = PromptTemplate(
                template="""
Answer using ONLY context.

Context:
{context}

Question:
{question}
""",
                input_variables=["context", "question"]
            )

            # Chain
            st.session_state.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            st.session_state.pdf_name = uploaded_file.name
            os.unlink(tmp_path)

        st.success("PDF Loaded!")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────
# MAIN CHAT
# ─────────────────────────────

st.title("💬 Chat With AI")

# Show messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    # Store user
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                if st.session_state.rag_chain:
                    result = st.session_state.rag_chain.invoke({
                        "query": user_input
                    })
                    answer = result["result"]
                else:
                    answer = llm.invoke(user_input).content

            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)

    # Store assistant
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })