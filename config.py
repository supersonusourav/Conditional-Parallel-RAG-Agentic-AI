import os
import mlflow
from mlflow.langchain import autolog
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Point explicitly to the running MLflow UI server
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("College_Assistant_RAG")

# Enable auto-tracing for LangChain/LangGraph
autolog()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0.4)