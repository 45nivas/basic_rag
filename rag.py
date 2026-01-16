"""
RAG Module: Document Loading, Chunking, Embeddings, and Retrieval
This module handles the core RAG functionality including document processing and retrieval.
"""

import os
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    """
    A simple RAG (Retrieval-Augmented Generation) implementation.
    Loads documents, chunks them, creates embeddings, and retrieves relevant chunks.
    """
    
    def __init__(self, data_folder: str = "data", chunk_size: int = 500):
        """
        Initialize the RAG system.
        
        Args:
            data_folder: Path to folder containing documents
            chunk_size: Number of characters per chunk
        """
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
        
        # Load a lightweight embedding model
        # Using sentence-transformers for simple and effective embeddings
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
        
    def load_documents(self) -> List[str]:
        """
        Load all text files from the data folder.
        
        Returns:
            List of document contents as strings
        """
        documents = []
        
        if not os.path.exists(self.data_folder):
            print(f"Warning: {self.data_folder} does not exist!")
            return documents
            
        # Read all .txt files in the data folder
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(content)
                        print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with some overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        # Add a small overlap (20% of chunk size) to maintain context
        overlap = self.chunk_size // 5
        
        # Split into chunks
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                
            start += self.chunk_size - overlap
            
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print("Embeddings created!")
        return embeddings
    
    def build_index(self):
        """
        Build the RAG index by loading documents, chunking, and creating embeddings.
        This prepares the system for retrieval.
        """
        print("\n=== Building RAG Index ===")
        
        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            print("No documents loaded!")
            return
            
        # Step 2: Chunk all documents
        print("\nChunking documents...")
        for doc in documents:
            doc_chunks = self.chunk_text(doc)
            self.chunks.extend(doc_chunks)
        print(f"Created {len(self.chunks)} chunks")
        
        # Step 3: Create embeddings for all chunks
        self.embeddings = self.create_embeddings(self.chunks)
        
        print("\n=== Index Built Successfully! ===\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for a given query.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            print("Index not built! Call build_index() first.")
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarity with all chunk embeddings
        # Cosine similarity = dot product of normalized vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis]
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
            
        return results
    
    def format_context(self, retrieved_chunks: List[Tuple[str, float]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            retrieved_chunks: List of (chunk_text, similarity_score) tuples
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant information found."
        
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Context {i}]:\n{chunk}")
            
        return "\n\n".join(context_parts)
