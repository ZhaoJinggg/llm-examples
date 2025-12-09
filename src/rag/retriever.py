import streamlit as st
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config

@st.cache_resource
def hybrid_retriever() -> PineconeHybridSearchRetriever:
    # 1. Embeddings Model (Dense Vector)
    embedding_model = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDINGS_MODEL)

    # 2. BM25 Encoder (Sparse Vector)
    bm25_encoder = BM25Encoder().default()

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