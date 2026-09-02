import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL_NAME = "qwen/qwen3.6-27b"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )


embeddings = get_embeddings()

llm = ChatGroq(
    model=LLM_MODEL_NAME,
    temperature=0.4
)
