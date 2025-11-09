import os
import streamlit as st
from src.rag_agent import RAGAgentService

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    
st.title("💬 RAG Chatbot")
st.caption("🚀 A Streamlit chatbot powered by RAG with knowledge base")

# Initialize RAG agent
@st.cache_resource
def get_rag_agent():
    """Initialize and cache RAG agent"""
    return RAGAgentService()

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
    
    # Get RAG agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                rag_agent = get_rag_agent()
                response = rag_agent.query(prompt)
                
                # Extract the final response from agent
                # The agent returns messages, get the last AI message
                if response and "messages" in response:
                    # Get the last assistant message
                    ai_messages = [m for m in response["messages"] if hasattr(m, 'type') and m.type == 'ai']
                    if ai_messages:
                        answer = ai_messages[-1].content
                    else:
                        answer = "I apologize, but I couldn't generate a response."
                else:
                    answer = "I apologize, but I couldn't generate a response."
                
                # Display and store response
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
