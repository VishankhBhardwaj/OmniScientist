import streamlit as st
from main import ChatBot

st.set_page_config(page_title="Research Assistant", page_icon="💡", layout="centered")

st.title("Research Assistant Chatbot 💡")
st.markdown("Your advanced AI assistant equipped with Web Search, Arxiv, Wikipedia, and Wolfram!")

# Initialize Chatbot and Session State
if "bot" not in st.session_state:
    try:
        # Load the backend chatbot
        st.session_state.bot = ChatBot()
    except Exception as e:
        st.error(f"Fail to initialize chatbot: {e}")
        st.stop()

if "messages" not in st.session_state:
    # Set default welcome message
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Research Assistant. How can I help you today?"}
    ]

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your research..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        with st.spinner("Researching and Thinking..."):
            try:
                # Call the backend chat function
                response = st.session_state.bot.chat(prompt)
                st.markdown(response)
                # Save assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error fetching response: {e}")
