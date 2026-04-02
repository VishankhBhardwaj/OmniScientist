# Research Assistant Chatbot 💡

An advanced AI assistant for academic research and paper analysis, built with [LangChain](https://python.langchain.com/), [Streamlit](https://streamlit.io/), and powered by the Groq API (LLaMA 3.3).

This application allows users to upload PDF research papers, store their embeddings in a Pinecone vector database, and perform both context-augmented document retrieval (Standard RAG) and external tool-assisted research (Agent Mode).

## Features

- **Standard RAG Mode (Documents Only):** Upload PDF documents, chunk them, embed them using HuggingFace (`all-MiniLM-L6-v2`), and query against your knowledge base with a history-aware retriever.
- **Agent Mode (Web + Tools):** Ask complex research questions. The chatbot can intelligently decide to use external tools to fetch up-to-date and broader information.
  - *Tools Available:* 
    - Wikipedia Search
    - ArXiv Search
    - Web Search (via Tavily)
    - Wolfram Alpha
    - Python Code Execution (REPL)
- **Interactive UI:** Built with Streamlit, featuring a sidebar for settings, file uploads, and a chat interface that retains conversation history.

## Project Structure

- `frontend.py`: Streamlit application script containing the user interface, sidebar settings, and chat logic.
- `main.py`: Core logic including the `ChatBot` class. Sets up the LangChain pipeline, embeddings, Pinecone vector store, prompts, RAG chains, and Agent executor.
- `tools.py`: Definition of all LangChain external tools (Wikipedia, ArXiv, Web Search, Wolfram Alpha, Python REPL).
- `requirements.txt`: Python package dependencies.

## Prerequisites

Ensure you have Python 3.8+ installed. You will also need API keys for the following services:
- [GroqAPI](https://console.groq.com/keys) for LLaMA 3.3
- [HuggingFace](https://huggingface.co/settings/tokens) for using the sentence-transformer embeddings
- [Pinecone](https://app.pinecone.io/) for the vector database
- [Tavily](https://tavily.com/) (Optional) for web search tool
- [Wolfram Alpha](https://developer.wolframalpha.com/) (Optional) for math/factual queries

## Installation

1. **Clone the repository** (if applicable) and navigate to the project directory:
   ```bash
   cd RAG PROJECT
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root of the project and add your API keys:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   HF_TOKEN="your_huggingface_token_here"
   PINECONE_API_KEY="your_pinecone_api_key_here"
   TAVILY_API_KEY="your_tavily_api_key_here"
   WOLFRAM_ALPHA_APPID="your_wolfram_alpha_appid_here"
   ```

5. **Pinecone Index Initialization:**
   The default code looks for a Pinecone index named `researchchatbot` using dimension `384` (for `all-MiniLM-L6-v2`). Make sure you create this index inside your Pinecone dashboard before running the app.

## Usage

You can run the application directly via Streamlit:

```bash
streamlit run frontend.py
```

### Modes of Operation:

1. **Upload Papers:** Use the sidebar to upload a PDF. It will be automatically parsed, split into chunks, and stored in your Pinecone index.
2. **Standard RAG Mode:** Ask questions directly related to the documents you've uploaded. The chatbot will retrieve the relevant chunks and answer based strictly on the context.
3. **Agent Mode:** If you ask general questions or need external data, the assistant can query Wikipedia, ArXiv, or use Web Search / Wolfram Alpha to compile an answer.
4. **Clear History:** Use the sidebar option to reset the chatbot's memory.

Alternatively, you can test the chatbot in the terminal by running:
```bash
python main.py
```
*(This will launch a simple CLI chat loop)*
