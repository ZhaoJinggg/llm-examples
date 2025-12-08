from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load Environment Variables
load_dotenv()

# Chat Model
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

# Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Vector Store
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "knowledge-base"
index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
retriever = vector_store.as_retriever()

## Load Documents
from langchain_community.document_loaders import PyPDFDirectoryLoader

directory_path = "./Documents/Introduction of Software Engineering"
loader = PyPDFDirectoryLoader(directory_path)

docs = loader.load()

print(f"Loaded {len(docs)} pages/documents from all PDFs in the directory.")

## Splitting Documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # chunk size (characters)
    chunk_overlap=150,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

doc_splits = text_splitter.split_documents(docs)

print(f"Split documents into {len(doc_splits)} sub-documents.")

## Index the chunks into vectorstore
document_ids = vector_store.add_documents(documents=doc_splits)

print(document_ids[:3])

from langchain.tools import tool
from tavily import TavilyClient

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
    
"""
@tool
def internet_search(query: str, max_results: int = 5) -> str:
    '''Search the web for information.'''
    results = tavily_client.search(query, max_results=max_results)
    return results

@wrap_tool_call
def handle_tool_errors(request, handler):
    '''Handle tool execution errors with custom messages.'''
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f'Tool error: Please check your input and try again. ({str(e)})',
            tool_call_id=request.tool_call["id"]
        )
"""

from langchain.agents import create_agent

tools = [retrieve_context]
# middleware=[handle_tool_errors]

prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt,
    # middleware=middleware
)

query = ("What is software engineering?")

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()