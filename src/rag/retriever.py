import streamlit as st
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

@st.cache_resource
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDINGS_MODEL,
        model_kwargs=Config.EMBEDDINGS_MODEL_KWARGS,
        encode_kwargs=Config.EMBEDDINGS_MODEL_ENCODE_KWARGS  
    )
    # Print device information
    try:
        # HuggingFaceEmbeddings uses sentence-transformers, which stores the model in .client
        device = embedding_model.client.device
        print(f"🧐 Embedding Model is using device: {str(device).upper()}")
    except Exception as e:
        print(f"🧐 Embedding Model device check failed: {e}")
        
    return embedding_model  

@st.cache_resource
def get_bm25_encoder():
    bm25_encoder = BM25Encoder.default()
    return bm25_encoder

def hybrid_retriever() -> PineconeHybridSearchRetriever:  
    # 1. Get Cached Embeddings Model (Dense Vector)
    embedding_model = get_embedding_model()
    
    # 2. Get Cached BM25 Encoder (Sparse Vector)
    bm25_encoder = get_bm25_encoder()

    # 3. Connect to Pinecone
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.PINECONE_INDEX_NAME)

    # 4. Build hybrid retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embedding_model,
        sparse_encoder=bm25_encoder,
        index=index,
        alpha=Config.RETRIEVER_ALPHA,
        top_k=Config.RETRIEVER_K
    )

    return retriever
