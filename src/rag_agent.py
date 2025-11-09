from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from src.vector_store import VectorStoreService

class RAGAgentService:
    """Service for RAG agent operations"""
    
    _instance = None
    _agent = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGAgentService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize RAG agent service (singleton pattern)"""
        if self._agent is None:
            self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the RAG agent with tools and model"""
        # Initialize vector store service
        self.vector_store_service = VectorStoreService()
        
        # Initialize chat model
        self.model = init_chat_model(
            "google_genai:gemini-2.5-flash",
            temperature=0.7,
            timeout=30,
            max_tokens=1000,
        )
        
        # Define retrieval tool
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information from knowledge base to help answer a query."""
            retrieved_docs = self.vector_store_service.similarity_search(query, k=5)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source_file', 'Unknown')}\n"
                 f"Page: {doc.metadata.get('page', 'N/A')}\n"
                 f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        self.tools = [retrieve_context]
        
        # Define system prompt
        prompt = (
            "You are a helpful AI assistant with access to a knowledge base. "
            "When answering questions, use the retrieve_context tool to search for relevant information. "
            "Always cite your sources by mentioning the document name and page number when available. "
            "If you cannot find relevant information in the knowledge base, say so clearly."
        )
        
        # Create agent
        self._agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=prompt,
        )
    
    def get_agent(self):
        """Get the RAG agent instance"""
        return self._agent
    
    def query(self, user_query):
        """
        Query the RAG agent
        
        Args:
            user_query: User's question
            
        Returns:
            Agent's response
        """
        try:
            response = self._agent.invoke({"messages": [("user", user_query)]})
            return response
        except Exception as e:
            raise Exception(f"Error querying RAG agent: {str(e)}")

