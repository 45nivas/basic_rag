# 🤖 Local RAG Chatbot

Hey! This is a simple chatbot that lets you "chat" with your own text documents. It runs completely locally on your machine using **Ollama**, so it's free and private.

## 🌟 What does it do?

You put your text files in a folder, and the bot reads them. When you ask a question, it finds the relevant parts of your files and uses a local AI model to answer you based _only_ on that info.

It understands abbreviations too! (e.g., if you ask about "AIML", it knows you mean Artificial Intelligence and Machine Learning).

## 🛠️ Requirements

1.  **Python** installed on your computer.
2.  **[Ollama](https://ollama.com/)** installed and running.

## 🚀 Quick Start

### 1. Get the Model
Open your terminal and run:
```bash
ollama pull llama3.2
```
Make sure Ollama is running in the background (`ollama serve`).

### 2. Install Python Libraries
```bash
pip install -r requirements.txt
```

### 3. Add Your Data
Drop any `.txt` files you want the bot to know about into the `data/` folder.

### 4. Run it!
```bash
streamlit run app.py
```
A browser window should pop up, and you can start chatting.

## 🧩 How it works (The simple version)

*   **app.py**: The website part (built with Streamlit).
*   **rag.py**: The brain. It reads your files, chops them into small pieces, and searches for the best pieces when you ask a question.
*   **Ollama**: The local AI that takes those pieces and writes a nice answer for you.

## 🐛 Having issues?
*   **"Connection refused"**: Is Ollama running? Open a terminal and type `ollama serve`.
*   **"No documents loaded"**: Did you put `.txt` files in the `data/` folder?

Enjoy chatting with your docs! 👋
