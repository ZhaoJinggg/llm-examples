from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from src.config import Config
from src.rag.vectorstore import get_vector_store

class Agent:
    def __init__(self):
        # Initialize Vector Store
        self.vector_store = get_vector_store()
        
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
            retrieved_docs = self.vector_store.similarity_search(query, k=Config.RETRIEVER_K)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            # Convert docs to a serializable artifact representation
            artifacts = [
                {"metadata": doc.metadata, "content": doc.page_content}
                for doc in retrieved_docs
            ]
            return serialized, artifacts
        
        # Initialize LangChain Agent
        self.agent = create_agent(
            model=self.model,
            tools=[retrieve_context],
            system_prompt=Config.SYSTEM_PROMPT,
        )