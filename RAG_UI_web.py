import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pickle

import os
from langchain_community.vectorstores import FAISS

def load_retriever():
    # Get absolute path to the faiss_index folder
    faiss_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    # Load the FAISS vector store from that path
    db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    return db.as_retriever()


# Load FAISS index and embedding model
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return retriever

# Load LLM via Ollama (no API key required)
@st.cache_resource
def load_llm():
    return Ollama(model="tinyllama")  # Change to 'mistral', 'llama2', etc. if needed

# Setup QA chain
@st.cache_resource
def setup_qa_chain():
    retriever = load_retriever()
    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Streamlit UI
st.set_page_config(page_title="Cybersecurity RAG Assistant", layout="wide")
st.title("🛡️ Cybersecurity Knowledge Assistant (RAG)")

question = st.text_input("Ask a cybersecurity question (e.g., What is lateral movement?)")

if question:
    with st.spinner("🔍 Retrieving answer..."):
        qa_chain = setup_qa_chain()
        result = qa_chain.run(question)
        st.success("✅ Answer:")
        st.write(result)
