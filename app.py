"""
Streamlit RAG Chatbot Application
A simple chatbot interface using RAG (Retrieval-Augmented Generation)
"""

import os
from dotenv import load_dotenv
import streamlit as st
import ollama
from rag import SimpleRAG

# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("A chatbot powered by Retrieval-Augmented Generation")


# Initialize the RAG system
@st.cache_resource
def initialize_rag():
    """
    Initialize and build the RAG index.
    This is cached so it only runs once when the app starts.
    """
    rag = SimpleRAG(data_folder="data", chunk_size=500)
    rag.build_index()
    return rag


# Initialize components
with st.spinner("Loading RAG system..."):
    rag = initialize_rag()

# Default Ollama model
OLLAMA_MODEL = "llama3.2"


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask me anything about the knowledge base..."):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Step 1: Retrieve relevant chunks from the knowledge base
            retrieved_chunks = rag.retrieve(prompt, top_k=3)
            
            # Step 2: Format the context
            context = rag.format_context(retrieved_chunks)
            
            # Step 3: Create the prompt for the LLM
            system_prompt = """You are a knowledgeable assistant. Your goal is to provide accurate information based on the context provided.

Instructions:
1. Always answer using only the provided Context.
2. If the Context does not contain the answer, respond with:
   "I could not find the answer in the provided documents."
3. Do not use external or general knowledge.
4. Do not guess or hallucinate.


Context:
{context}
"""
            
            # Step 4: Call the LLM
            try:
                # Generate response with Ollama
                response = ollama.chat(model=OLLAMA_MODEL, messages=[
                    {'role': 'system', 'content': system_prompt.format(context=context)},
                    {'role': 'user', 'content': prompt},
                ])
                
                # Extract the response
                assistant_message = response['message']['content']
                
                # Display the response
                st.markdown(assistant_message)
                
                # Show retrieved context in an expander (optional, for transparency)
                with st.expander("📚 View Retrieved Context"):
                    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                        st.markdown(f"**Chunk {i}** (similarity: {score:.3f})")
                        st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                        st.divider()
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                assistant_message = "Sorry, I encountered an error. Please try again."
                st.markdown(assistant_message)
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message
            })


# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a simple RAG (Retrieval-Augmented Generation) chatbot.
    
    **How it works:**
    1. Your question is converted into an embedding
    2. Similar text chunks are retrieved from the knowledge base
    3. The context is sent to an LLM (Ollama)
    4. The LLM generates a response based on the context
    
    **Knowledge Base:**
    - Documents are loaded from the `data/` folder
    - Currently using sample AI/ML knowledge
    """)
    
    st.divider()
    
    st.markdown(f"**Total chunks indexed:** {len(rag.chunks)}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
