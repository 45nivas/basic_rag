# 🤖 RAG Chatbot

A simple chatbot built with **Retrieval-Augmented Generation (RAG)** using Python and Streamlit.

## 📚 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances Large Language Models (LLMs) by providing them with relevant context from a knowledge base.

**How it works:**

1. **Document Ingestion**: Load documents and split them into chunks
2. **Embeddings**: Convert text chunks into numerical vectors (embeddings)
3. **Retrieval**: When a user asks a question, find the most similar chunks using vector similarity search
4. **Generation**: Send the retrieved context + user question to an LLM to generate an informed response

This approach allows the LLM to answer questions based on your specific documents, rather than just its training data.

## 🏗️ Project Structure

```
project/
│
├── app.py                    # Streamlit UI and chat logic
├── rag.py                    # RAG implementation (loading, chunking, retrieval)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── data/                     # Your knowledge base documents
    └── sample_knowledge.txt  # Sample document about AI/ML
```

## ⚙️ How It Works

### 1. **rag.py** - RAG Core Logic

- **Document Loading**: Reads all `.txt` files from the `data/` folder
- **Chunking**: Splits documents into 500-character chunks with 20% overlap
- **Embeddings**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to create vector embeddings
- **Retrieval**: Finds top-k most similar chunks using cosine similarity

### 2. **app.py** - Streamlit Interface

- Provides a chat interface with message history
- On each user query:
  1. Retrieves relevant chunks from the knowledge base
  2. Formats them as context
  3. Sends context + question to Ollama (running locally)
  4. Displays the response
- Shows retrieved context for transparency

## 🚀 How to Run the Project

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and running locally
- Pull the model: `ollama pull llama3.2`

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` - Web UI framework
- `ollama` - Python client for Ollama
- `sentence-transformers` - For creating embeddings
- `numpy` - For vector operations
- `torch` - Required by sentence-transformers

### Step 2: Start Ollama

Ensure Ollama is running in the background.

```bash
ollama serve
# In another terminal:
ollama pull llama3.2
```

### Step 3: Add Your Documents

Place your text files (`.txt`) in the `data/` folder. The project includes a sample file `sample_knowledge.txt` about AI/ML topics.

**To use your own documents:**
- Add `.txt` files to the `data/` folder
- Restart the app to reindex

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will:
1. Load and index all documents from `data/`
2. Open in your browser (usually at http://localhost:8501)
3. Be ready to chat with **local Ollama models**!

## 💬 Using the Chatbot

1. Type your question in the chat input
2. The system will:
   - Search for relevant information in your documents
   - Generate a response using **Ollama (Llama 3.2)**
   - Show you which context was used (click "View Retrieved Context")
3. Chat history is maintained during your session
4. Click "Clear Chat History" in the sidebar to start fresh

## 🔧 Customization

### Change the LLM

In `app.py`, modify the model:

```python
# Change the model name (ensure you pulled it with `ollama pull modelname`)
OLLAMA_MODEL = "mistral"
```

### Adjust Chunk Size

In `app.py`, change the initialization:

```python
rag = SimpleRAG(data_folder="data", chunk_size=1000)  # default is 500
```

### Change Number of Retrieved Chunks

In `app.py`, modify:

```python
retrieved_chunks = rag.retrieve(prompt, top_k=5)  # default is 3
```

## 🐛 Troubleshooting

**"Connection refused" or Ollama errors**
- Ensure Ollama is installed and running (`ollama serve`)
- Check if you have the model pulled (`ollama list`)

**"No documents loaded!"**
- Verify that `.txt` files exist in the `data/` folder
- Check file permissions

**Model download takes time**
- First run downloads the embedding model (~80MB)
- Subsequent runs are faster

**Out of memory**
- Reduce chunk size or number of documents
- Use a smaller embedding model

## 📝 Notes

- This is a **basic implementation** for learning purposes
- Embeddings are stored in memory (lost when app restarts)
- No persistent database - suitable for small knowledge bases
- For production use, consider:
  - Persistent vector database (Pinecone, Weaviate, ChromaDB)
  - Caching mechanisms
  - Error handling and logging
  - User authentication

## 🎯 Assignment Completion Checklist

✅ Working Streamlit UI with chat input and history  
✅ Basic RAG flow: ingestion → retrieval → response  
✅ Local text file as data source  
✅ Embeddings + vector search (using cosine similarity)  
✅ LLM integration (Ollama - Local!)  
✅ Clean, readable code with comments  
✅ Simple project structure  
✅ Complete README with system explanation and setup instructions

---

Built as part of an intern technical assessment. Focused on clarity and basic RAG implementation.
