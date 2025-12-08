import streamlit as st
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.config import Config

@st.cache_resource
def get_vector_store() -> PineconeVectorStore:
    # Embeddings Model (Google Generative AI)
    embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDINGS_MODEL)
    
    # Vector Store Model (Pinecone)
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    
    # Return the vector store
    return vector_store

def index_documents(chunks: List[Document]) -> List[str]:
    # Initialize the vector store
    vector_store = get_vector_store()
    
    # Index the documents
    ids = vector_store.add_documents(documents=chunks)
    
    # Return the IDs of the added documents
    return ids

def delete_documents(ref_ids: List[str]) -> None:
    """Remove all vectors whose metadata ref_id matches the provided ids."""
    # Deduplicate and ignore empty identifiers
    unique_ids = sorted({ref_id for ref_id in ref_ids if ref_id})
    if not unique_ids:
        return

    vector_store = get_vector_store()
    try:
        vector_store.delete(filter={"ref_id": {"$in": unique_ids}})
    except Exception as exc:
        raise RuntimeError(f"Failed to delete vectors for ref_ids {unique_ids}") from exc