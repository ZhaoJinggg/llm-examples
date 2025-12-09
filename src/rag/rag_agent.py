from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from src.config import Config
from src.rag.retriever import hybrid_retriever
from src.rag.reranker import rerank_documents

class Agent:
    def __init__(self):
        # Initialize Hybrid Retriever
        self.retriever = hybrid_retriever()
        
        # Initialize Chat Model
        self.model = init_chat_model(
            model=Config.CHAT_MODEL_NAME,
            temperature=Config.CHAT_MODEL_TEMPERATURE,
            timeout=Config.CHAT_MODEL_TIMEOUT,
            max_tokens=Config.CHAT_MODEL_MAX_TOKENS,
        )

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            # Invoke Hybrid Search
            retrieved_docs = self.retriever.invoke(query)

            # Rerank Documents
            reranked_docs = rerank_documents(query, retrieved_docs, top_n=Config.RERANKER_TOP_N)
            
            # Convert docs to a serializable artifact representation
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in reranked_docs
            )
            return serialized, reranked_docs
        
        # Initialize LangChain Agent
        self.agent = create_agent(
            model=self.model,
            tools=[retrieve_context],
            system_prompt=Config.SYSTEM_PROMPT,
        )