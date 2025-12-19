# 🤖 RAG Chatbot with Multi-Agent System

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, featuring a multi-agent architecture powered by LangGraph. The system combines internal knowledge base retrieval with real-time web search capabilities.

## ✨ Features

- **🧠 Multi-Agent Architecture**: Supervisor agent orchestrates specialized knowledge base and web search agents
- **🔍 Hybrid Retrieval**: Combines dense vector embeddings (BGE-M3) with sparse BM25 for optimal search results
- **📊 Document Reranking**: Uses BGE reranker model to improve retrieval accuracy
- **📚 Knowledge Base Management**: Upload, manage, and delete documents through a web interface
- **🌐 Real-time Web Search**: Integrated Tavily search for up-to-date information
- **💾 Conversation Persistence**: PostgreSQL-backed conversation state management
- **📄 Multi-format Support**: Supports PDF, TXT, MD, DOCX, and CSV files

## 🏗️ Architecture

```
Supervisor Agent
├── Knowledge Base Agent (retrieves from internal documents)
└── Web Search Agent (searches the internet via Tavily)
```

The system uses:
- **Dense Vectors**: HuggingFace BGE-M3 embeddings stored in Pinecone
- **Sparse Vectors**: BM25 encoding for keyword-based search
- **Reranking**: BGE reranker-v2-m3 for relevance scoring
- **LLM**: Google Gemini 2.5 Flash for generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL database (for conversation state)
- Pinecone account and API key
- Google API key (for Gemini)
- Tavily API key (for web search)
- Firebase account (for document storage, optional)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llm-examples
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
DB_URI=postgresql://user:password@host:port/database
LANGSMITH_API_KEY=your_langsmith_api_key  # Optional, for tracing
```

4. Configure Firebase (for knowledge base management):
   - Add Firebase credentials to Streamlit secrets or environment variables

5. Run the application:
```bash
streamlit run Chatbot.py
```

## 📁 Project Structure

```
.
├── Chatbot.py                 # Main Streamlit chat interface
├── pages/
│   └── Knowledge_Base.py      # Document management page
├── src/
│   ├── config.py              # Configuration settings
│   ├── firebase_init.py       # Firebase initialization
│   └── rag/
│       ├── rag_agent.py       # Multi-agent system
│       ├── retriever.py       # Hybrid retrieval setup
│       ├── reranker.py        # Document reranking
│       ├── data_loader.py     # Document loading utilities
│       ├── text_splitter.py   # Text chunking
│       └── vectorstore.py     # Vector store operations
└── requirements.txt           # Python dependencies
```

## 🔧 Configuration

Key settings can be modified in `src/config.py`:

- **Model**: `CHAT_MODEL_NAME` (default: `google_genai:gemini-2.5-flash`)
- **Embeddings**: `EMBEDDINGS_MODEL` (default: `BAAI/bge-m3`)
- **Reranker**: `RERANKER_MODEL` (default: `BAAI/bge-reranker-v2-m3`)
- **Chunk Size**: `TEXT_SPLITTER_CHUNK_SIZE` (default: 400)
- **Retrieval**: `RETRIEVER_K` (default: 20), `RETRIEVER_ALPHA` (default: 0.7)

## 📖 Usage

1. **Chat Interface** (`Chatbot.py`):
   - Enter your Google API key in the sidebar
   - Ask questions about your knowledge base or general topics
   - The supervisor agent automatically routes queries to the appropriate sub-agent

2. **Knowledge Base Management** (`pages/Knowledge_Base.py`):
   - Upload documents (PDF, TXT, MD, DOCX, CSV)
   - View, download, or delete uploaded files
   - Documents are automatically indexed to Pinecone

## 🚢 Deployment

### Recommended: Streamlit Cloud

**Streamlit Cloud** is the recommended deployment platform for this application.

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Configure secrets in Streamlit Cloud dashboard:
   ```
   GOOGLE_API_KEY=your_key
   PINECONE_API_KEY=your_key
   TAVILY_API_KEY=your_key
   DB_URI=postgresql://...
   LANGSMITH_API_KEY=your_key  # Optional
   ```
5. For Firebase, add credentials to Streamlit secrets as JSON

**Note:** Streamlit Cloud has resource limits. For heavy ML workloads (reranker model), consider:
- Using CPU-only models for deployment
- Or deploying to platforms with GPU support (see alternatives below)

### Alternative Deployment Options

**Other platforms that support Streamlit:**
- **Heroku** (with buildpacks)
- **AWS EC2/ECS** (with Docker)
- **Google Cloud Run** (containerized)
- **Azure Container Instances**
- **DigitalOcean App Platform**

### LangSmith (Optional Monitoring)

**LangSmith** is NOT a deployment platform, but an optional monitoring tool:

- **Purpose**: Track LLM calls, debug prompts, monitor performance
- **Setup**: 
  1. Sign up at [smith.langchain.com](https://smith.langchain.com)
  2. Get your API key
  3. Set `LANGSMITH_TRACING=true` in environment variables
  4. Configure `LANGSMITH_PROJECT` and `LANGSMITH_ENDPOINT` if needed

**Benefits:**
- Monitor agent decision-making
- Debug retrieval quality
- Track token usage and costs
- Analyze user queries and responses

## 🛠️ Technologies

- **Frontend**: Streamlit
- **LLM Framework**: LangChain, LangGraph
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace Transformers
- **LLM**: Google Gemini
- **Search**: Tavily
- **Database**: PostgreSQL (conversation state), Firebase (document storage)
- **ML**: PyTorch, Transformers

## 📝 License

See LICENSE file for details.
