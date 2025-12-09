import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

    # LangSmith Configuration
    LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")
    LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")

    # Model Configuration 
    CHAT_MODEL_NAME = "google_genai:gemini-2.5-flash"
    CHAT_MODEL_TEMPERATURE = 0.7
    CHAT_MODEL_TIMEOUT = 30
    CHAT_MODEL_MAX_TOKENS = 1000

    # Embeddings Model Configuration
    EMBEDDINGS_MODEL = "models/gemini-embedding-001"

    # Pinecone Index Configuration
    # PINECONE_INDEX_NAME = "knowledge-base"
    PINECONE_INDEX_NAME = "hybrid-search-index"

    # Text Splitter Configuration
    TEXT_SPLITTER_CHUNK_SIZE = 400
    TEXT_SPLITTER_CHUNK_OVERLAP = 50
    TEXT_SPLITTER_ADD_START_INDEX = True
    
    # Configure Reranker model
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_N = 3

    # Retriever Configuration 
    RETRIEVER_ALPHA = 0.7
    RETRIEVER_K = 20 

    # System Prompt
    SYSTEM_PROMPT = (
        "You are an AI assistant that helps users to find accurate information. "
        "You can answer questions, provide explanations, and generate text based on the input. "
        "Please answer the user's question exactly in the same language as the question or follow user's instructions. "
        "If you don't know the answer, please reply the user that you don't know. "
        "If you need more information, you can ask the user for clarification. "
        "You have access to a tool that retrieves context from knowledge base. "
        "Use the tool to help answer user queries."
        "Please be professional to the user."
    )