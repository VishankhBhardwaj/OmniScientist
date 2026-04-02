import streamlit as st
import os
from main import ChatBot

# Page Configuration
st.set_page_config(page_title="Research Assistant", page_icon="💡", layout="wide")

st.title("Research Assistant Chatbot 💡")
st.markdown("Your advanced AI assistant for academic research and paper analysis.")

# Initialize Chatbot and Session State
if "bot" not in st.session_state:
    try:
        with st.spinner("Initializing ResearchAI Engine..."):
            st.session_state.bot = ChatBot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Research Assistant. How can I help you with your research today?"}
    ]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Search Mode
    search_mode = st.radio(
        "Search Mode",
        ["Standard RAG (Documents Only)", "Agent Mode (Web + Tools)"],
        help="Standard RAG focuses strictly on your uploaded papers. Agent Mode uses external tools like Web, Arxiv, and Wikipedia."
    )
    
    st.divider()
    
    # 2. File Uploader
    st.header("📄 Upload Research Papers")
    uploaded_file = st.file_uploader("Upload a PDF to your Pinecone index", type="pdf")
    if uploaded_file is not None:
        if st.button("Process & Store Document"):
            with st.spinner("Processing PDF and updating Vector DB..."):
                try:
                    # Save temporary file
                    temp_path = os.path.join("tmp", uploaded_file.name)
                    os.makedirs("tmp", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Call ChatBot functions
                    docs = st.session_state.bot.load_document(temp_path)
                    splits = st.session_state.bot.split_document(docs)
                    st.session_state.bot.embed_and_store(splits)
                    
                    st.success(f"Successfully added '{uploaded_file.name}' to your knowledge base!")
                    # Clean up
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    st.divider()

    # 3. Clear Chat History
    if st.button("🗑️ Clear Chat History"):
        st.session_state.bot.clear_history()
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you now?"}
        ]
        st.rerun()

# --- CHAT INTERFACE ---

# Display all messages from session status
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Loop
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                # Select bot function based on mode
                if search_mode == "Standard RAG (Documents Only)":
                    response = st.session_state.bot.chat(prompt)
                else:
                    response = st.session_state.bot.chat_with_tools(prompt)
                
                st.markdown(response)
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
