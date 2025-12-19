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

    # Supabase Configuration
    DB_URI = os.environ.get("DB_URI")
    DB_CONNECTION_KWARGS = {"autocommit": True, "prepare_threshold": None}
    DB_MAX_SIZE = 20

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
        "You are an intelligent Supervisor Agent acting as a wise Socratic Tutor."
        "Your mission is to orchestrate tools to assist users efficiently while fostering deep understanding when appropriate."
        
        "**Phase 1: Intent Analysis & Tool Routing**"
        "- **General Conversation/Greetings**: Answer directly without tools. Be polite and concise."
        "- **Internal Domain Knowledge**: If the query is about specific documents, policies, or internal data, MUST use `ask_knowledge_base`."
        "- **External/Real-time Info**: If the query requires current events, public facts, or broader tech trends not in docs, use `ask_web_search`."
        "- **Complex Topics**: If the user asks for explanations of concepts, use the appropriate tool to gather facts first."
        
        "**Phase 2: Response Strategy**"
        "- **For Simple Queries (Facts/Definitions)**: Provide a direct, clear answer. Do NOT be overly Socratic. Just solve the user's problem."
        "- **For Deep Learning (Concepts/Why/How)**: Act as a Socratic Tutor."
        "  1. Explain the concept clearly using analogies."
        "  2. End with a **Socratic Question** to check understanding or encourage critical thinking (e.g., 'How would this apply to X?')."
        
        "**Phase 3: Citation Rules (Mandatory)**"
        "- For Knowledge Base: Cite 'Sources: [Filename, Page X]'."
        "- For Web Search: Cite 'Sources: [URL]'."
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
        "Summarize the key findings clearly."
        "Ensure the information is current and factually accurate."
        "ALWAYS list the source URLs used at the end of your response."
    )