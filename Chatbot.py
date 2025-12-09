import os
import streamlit as st
from src.rag.rag_agent import Agent
from src.rag.reranker import get_reranker_model

st.title("💬 RAG Chatbot")
st.caption("🚀 A Streamlit chatbot powered by RAG with knowledge base")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
 
# Initialize Reranker model
with st.spinner("Loading Reranker Model..."):
    get_reranker_model()

@st.cache_resource(show_spinner="Loading Agent Model...")
def get_agent():
    return Agent()

# Initialize RAG Agent
rag_agent = get_agent()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can answer questions based on your knowledge base. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:   
                # Invoke the agent
                response = rag_agent.agent.invoke({"messages": st.session_state.messages})
                # Normalize response content across possible return shapes
                if isinstance(response, dict):
                   # Common LangChain shapes
                    if "content" in response:
                        answer = response["content"]
                    elif "messages" in response and isinstance(response["messages"], list) and response["messages"]:
                        last = response["messages"][-1]
                        answer = getattr(last, "content", None) or last.get("content", "")
                    else:
                        answer = str(response)
                else:
                    answer = str(response)
                
                # Display and store response
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
