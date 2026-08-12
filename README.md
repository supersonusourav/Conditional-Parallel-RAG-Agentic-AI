# 🎓 Conditional Parallel RAG Agentic AI

An intelligent, multi-agent RAG system that dynamically classifies student queries, routes them in parallel to specialized knowledge bases (Academics & Fee Structures), and synthesizes context-aware answers using Groq-accelerated Llama 3.3.

---

## 🌟 Key Features

* **Agentic Query Classification:** Utilizes an LLM classifier node to break down incoming queries into specific intent categories (`academic`, `fee`, or `general`).
* **Conditional Parallel Retrieval:** Leverages LangGraph's fan-out execution to search independent FAISS vector stores concurrently for multi-intent questions.
* **Streamlit Olive UI:** Custom-themed, interactive chat interface with program-aware context management (`BCA`, `BBA`, `B.Com`).
* **Optimized Execution:** Employs `@st.cache_resource` for instant cold-starts and low-latency response generation.
* **MLflow Evaluation Support:** Integrated evaluation pipelines for tracking model faithfulness and response quality metrics.

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph, LangChain
* **LLM Engine:** ChatGroq (`llama-3.3-70b-versatile`)
* **Vector Store & Embeddings:** FAISS, Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Frontend:** Streamlit
* **Evaluation & Experiment Tracking:** MLflow
