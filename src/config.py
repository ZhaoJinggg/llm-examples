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
    RETRIEVER_K = 20

    # Agent Prompt
    SUPERVISOR_PROMPT = (
        "You are an intelligent Supervisor Agent orchestrating tools to assist users."
        "Analyze the user's intent to decide the best approach."
        "If the query is a general greeting, conversational, or solvable with your general knowledge, answer directly without calling tools."
        "Use the `ask_knowledge_base` tool for questions about internal documents or specific domain knowledge."
        "Use the `ask_web_search` tool for current events, public facts, or broader topics."
        "Prioritize the `ask_knowledge_base` tool if the query seems relevant to internal context."
        "Synthesize gathered information into a professional, clear, and helpful response."
        "Include the 'Sources' citation and references [file name and page number] at the end of final response."
    )

    KNOWLEDGE_AGENT_PROMPT = (
        "You are a specialized Knowledge Base Agent answering strictly based on retrieved context."
        "Use the `retrieve_context` tool to find relevant documents."
        "Your answers must be grounded ONLY in the provided documents; do not hallucinate or use outside knowledge."
        "List all unique sources used at the end of your response (include filename and page number)."
        "If the information is not in the context, state that the documents do not contain the answer."
    )

    SEARCH_AGENT_PROMPT = (
        "You are a specialized Web Search Agent providing up-to-date information from the internet."
        "Use the `web_search` tool to gather diverse and reliable sources."
        "Synthesize multiple results into a coherent, well-structured answer."
        "Avoid simply listing links; instead, summarize the key findings clearly."
        "Ensure the information is current and factually accurate."
    )