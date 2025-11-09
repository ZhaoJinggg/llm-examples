import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

class VectorStoreService:
    """Service for managing vector store operations"""
    
    _instance = None
    _vector_store = None
    _embeddings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize vector store service (singleton pattern)"""
        if self._vector_store is None:
            self._initialize_store()
    
    def _initialize_store(self):
        """Initialize Pinecone vector store and embeddings"""
        # Initialize embeddings model
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(name="knowlegde-base")
        
        # Create vector store
        self._vector_store = PineconeVectorStore(
            embedding=self._embeddings, 
            index=index
        )
    
    def get_vector_store(self):
        """Get the vector store instance"""
        return self._vector_store
    
    def get_embeddings(self):
        """Get the embeddings model"""
        return self._embeddings
    
    def add_documents(self, documents):
        """Add documents to vector store"""
        return self._vector_store.add_documents(documents=documents)
    
    def similarity_search(self, query, k=5):
        """Search for similar documents"""
        return self._vector_store.similarity_search(query, k=k)
    
    def delete_by_metadata(self, metadata_filter):
        """Delete documents by metadata filter"""
        # Note: This requires the vector store to support deletion
        # You may need to implement this based on your Pinecone setup
        pass

