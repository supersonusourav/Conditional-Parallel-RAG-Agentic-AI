import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import embeddings

def get_or_create_retriever(pdf_path: str, index_folder_name: str):
    """Loads FAISS index from disk if present; builds and saves it locally otherwise."""
    
    if os.path.exists(index_folder_name):
        # Load pre-built vectorstore index from disk
        vectorstore = FAISS.load_local(
            index_folder_name, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        # Load PDF and split text
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        # Build vector store and persist locally
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_folder_name)
    
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Persistent Retrievers
academic_retriever = get_or_create_retriever("academics_handbook.pdf", "faiss_academic_index")
fee_retriever = get_or_create_retriever("fee_structure.pdf", "faiss_fee_index")