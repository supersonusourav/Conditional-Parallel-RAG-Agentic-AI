import os
import mlflow
from mlflow.langchain import autolog
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Set up MLflow tracking (Fallback to local file store if server is unreachable)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("College_Assistant_RAG")
    autolog()
except Exception as e:
    # Disable remote tracking if server is unavailable (e.g. on Streamlit Cloud)
    print(f"MLflow tracking disabled or unreachable at {MLFLOW_URI}: {e}")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
#LLM_MODEL_NAME = "llama-3.3-70b-versatile"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0.4)
