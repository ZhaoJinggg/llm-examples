import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

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
    EMBEDDINGS_MODEL = "BAAI/bge-m3"
    EMBEDDINGS_MODEL_ENCODE_KWARGS = {'normalize_embeddings': True}
    EMBEDDINGS_MODEL_KWARGS = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    
    # Pinecone Index Configuration
    PINECONE_INDEX_NAME = "knowledge-base"
    #PINECONE_INDEX_NAME = "hybrid-search-index"

    # Text Splitter Configuration
    TEXT_SPLITTER_CHUNK_SIZE = 400
    TEXT_SPLITTER_CHUNK_OVERLAP = 50
    TEXT_SPLITTER_ADD_START_INDEX = True
    
    # Configure Reranker model
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_N = 5

    # Retriever Configuration 
    RETRIEVER_ALPHA = 0.7
    RETRIEVER_K = 10

    # Agent Prompt
    SUPERVISOR_PROMPT = (
        "You are an intelligent Supervisor Agent responsible for accurate information retrieval and synthesis. "
        "Your primary role is to understand user queries and route them to the most appropriate specialized tool."
        "You have two available tools: 1. ask_knowledge_base and 2. ask_web_search."
        "ask_knowledge_base: Use this for questions regarding internal documents, specific domain knowledge, or private data stored in the system. "
        "Prioritize this tool if the query seems relevant to internal context."
        "ask_web_search: Use this for current events, public knowledge, general facts, or when the knowledge base is insufficient."
        "Analyze First: Determine intent. Is it internal/specific (Knowledge Base) or external/general (Web Search)?"
        "Multi-step Reasoning: If a query requires both internal context and external verification, you may coordinate both tools."
        "Language Consistency: ALWAYS reply in the same language as the user's query."
        "Synthesis: Provide a coherent, well-structured final answer combining insights from tool outputs. Do not simply dump raw data."
        "Honesty: If neither tool provides a sufficient answer, clearly state what is missing and ask for clarification."
        "Tone: Maintain a helpful, professional, and concise persona."
    )

    KNOWLEDGE_AGENT_PROMPT = (
        "You are a specialized Knowledge Base Agent. Your goal is to answer questions based on the retrieved context documents."
        "Use the `retrieve_context` tool to find relevant information."
        "Base your answer on the provided documents."
        "If possible, mention which document or section the information comes from."
    )

    SEARCH_AGENT_PROMPT = (
        "You are a specialized Web Search Agent. Your goal is to find up-to-date and accurate information from the internet."
        "Use the `web_search` tool to gather information."
        "Synthesize multiple search results to provide a complete answer."
        "Summarize complex topics simply and clearly."
    )