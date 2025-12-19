import streamlit as st
import os
import uuid
from src.config import Config
from src.rag.rag_agent import Agent
from src.rag.reranker import get_reranker_model

st.title("💬 RAG Chatbot")
st.caption("🚀 A Streamlit chatbot powered by RAG with knowledge base")

with st.sidebar:
    google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")
    "[Get a Google API key](https://makersuite.google.com/app/apikey)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

# Initialize Reranker model
with st.spinner("Loading Reranker Model..."):
    get_reranker_model()

@st.cache_resource(show_spinner="Loading Agent Model...")
def get_agent(api_key=None):
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    elif Config.GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
    return Agent()

# Initialize RAG Agent
rag_agent = get_agent(google_api_key)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can answer questions based on your knowledge base. How can I help you?"}
    ]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Get assistant response with streaming
    with st.chat_message("assistant"):
        try:
            final_answer = ""
            
            # Use st.status to show agent progress
            with st.status("Thinking...", expanded=True) as status:
                
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                
                # Stream agent updates
                for chunk in rag_agent.agent.stream(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                    stream_mode="updates",
                ):
                    for step, data in chunk.items():
                        # Skip if data is None
                        if not data:
                            continue

                        # Get the last message from this step
                        if "messages" not in data or not data["messages"]:
                            continue
                            
                        last_message = data["messages"][-1]
                        
                        # Handle tool calls (agent deciding to use a tool)
                        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                tool_name = tool_call.get("name", "unknown")
                                st.write(f"🔧 Calling tool: `{tool_name}`")
                        
                        # Handle tool responses
                        elif step == "tools":
                            tool_name = getattr(last_message, "name", "tool")
                            st.write(f"✅ `{tool_name}` returned results")
                        
                        # Handle model responses (final answer)
                        elif step == "model":
                            content = getattr(last_message, "content", None)
                            if content:
                                # Check if this is a final response (no tool calls)
                                if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                                    st.write("✨ Generating response...")
                                    
                                    # Extract text content inline
                                    if isinstance(content, str):
                                        final_answer = content
                                    elif isinstance(content, list):
                                        text_parts = []
                                        for part in content:
                                            if isinstance(part, dict) and part.get("type") == "text":
                                                text_parts.append(part.get("text", ""))
                                            elif isinstance(part, str):
                                                text_parts.append(part)
                                        final_answer = "".join(text_parts)
                                    else:
                                        final_answer = str(content)
                
                status.update(label="✅ Complete!", state="complete", expanded=False)
            
            # Display final answer
            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                st.warning("No response generated.")
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})