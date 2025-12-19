from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
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

        # Define Tools
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information from the knowledge base to help answer a query."""
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
        
        @tool
        def web_search(query: str):
            """Search the web for information using Tavily."""
            # Instantiate Tavily tool (requires TAVILY_API_KEY in env)
            tool = TavilySearchResults(max_results=5)
            # Return results
            return tool.invoke(query)

        # Specialized Sub-Agents
        knowledge_agent = create_agent(
            model=self.model,
            tools=[retrieve_context],
            system_prompt=Config.KNOWLEDGE_AGENT_PROMPT
        )
        
        search_agent = create_agent(
            model=self.model,
            tools=[web_search],
            system_prompt=Config.SEARCH_AGENT_PROMPT
        )

        # Wrap Sub-Agents as Supervisor Tools
        @tool
        def ask_knowledge_base(query: str):
            """Use this tool to ask questions that might be found in the internal knowledge base documents.
            Input should be a standalone query string."""
            # Invoking the sub-agent
            result = knowledge_agent.invoke({"messages": [{"role": "user", "content": query}]})
            # Return the final response text
            if "messages" in result:
                return result["messages"][-1].content
            return str(result)

        @tool
        def ask_web_search(query: str):
            """Use this tool to search for information on the public web.
            Input should be a standalone query string."""
            # Invoking the sub-agent
            result = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
            # Return the final response text
            if "messages" in result:
                return result["messages"][-1].content
            return str(result)

        # Supervisor Agent 
        self.agent = create_agent(
            model=self.model,
            tools=[ask_knowledge_base, ask_web_search],
            system_prompt=Config.SUPERVISOR_PROMPT,
        )
