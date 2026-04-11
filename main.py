from dotenv import load_dotenv
load_dotenv()

import os
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from pinecone import Pinecone
from tools import web_search, arxiv_search, wikipedia_search, wolfram_alpha, python_commands
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


class ChatBot:
    def __init__(self):
        # Initialize core model
        self.model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=groq_api_key)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Connect to Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("researchchatbot")
        self.vectorstore = PineconeVectorStore(index=index, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Chat history for memory
        self.chat_history = []

        # Tools list
        self.tools = [web_search, arxiv_search, wikipedia_search, wolfram_alpha, python_commands]

        # ---- Prompt for history-aware retriever ----
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # ---- Main RAG Prompt ----
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are "ResearchAI", an expert research assistant. Your goal is to analyze the provided context from research papers/documents and answer the user's question with 100% accuracy.

### 📜 CONTEXT FROM DOCUMENTS:
{context}

### 🛠 INSTRUCTIONS:
1. **Strict Grounding:** Answer based on the provided context. Use tools for information not in context.
2. **Handle Uncertainty:** If the answer is not in context, say: "I'm sorry, but the provided documents do not contain information to answer this specific question."
3. **Citations:** When you quote or reference a point, mention the source/page number if available in the metadata.
4. **Structure:** Use bullet points and bold text for key terms to make the research findings easy to read.
5. **Tone:** Maintain a professional, academic, and objective tone."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # ---- Agent Prompt (for tools) ----
        self.agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are "ResearchAI", a powerful research assistant with access to tools.
Use the tools available to answer any research question accurately.
Always cite your sources when possible."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Build the History-Aware RAG Chain
        self.history_aware_retriever = create_history_aware_retriever(
            self.model, self.retriever, self.contextualize_q_prompt
        )
        self.document_chain = create_stuff_documents_chain(self.model, self.rag_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.document_chain)

        # Build the Agent (with tools)
        agent = create_tool_calling_agent(self.model, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def load_document(self, pdf_path: str):
        """Load a PDF document"""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return docs

    def split_document(self, docs):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)

    def embed_and_store(self, docs):
        """Embed and store documents in Pinecone"""
        self.vectorstore.add_documents(docs)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def chat(self, query: str) -> str:
        """Main chat method with RAG + Memory"""
        response = self.rag_chain.invoke({
            "input": query,
            "chat_history": self.chat_history
        })
        answer = response.get("answer", "I couldn't find an answer.")

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return answer

    # def chat_with_tools(self, query: str) -> str:
        """Chat using Agent with external tools (web search, arxiv, etc.)"""
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": self.chat_history
        })
        answer = response.get("output", "I couldn't find an answer.")

        # Update chat history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return answer

    def clear_history(self):
        """Clear conversation memory"""
        self.chat_history = []


if __name__ == "__main__":
    chatbot = ChatBot()
    print("ResearchAI is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        response = chatbot.chat(query)
        print(f"\nBot: {response}")