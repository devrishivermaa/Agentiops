# RAG/query_system.py
# RAG query system using FAISS vector store

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI

# -------------------------------------
# CONFIG
# -------------------------------------
VECTOR_DIR = "./vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# RAG settings
TOP_K = 5  # Number of chunks to retrieve
CONTEXT_LIMIT = 4000  # Max characters for context


# -------------------------------------
# RAG SYSTEM
# -------------------------------------

class RAGSystem:
    def __init__(self, vector_dir: str = VECTOR_DIR, model: Optional[str] = None):
        """Initialize RAG system with vector store."""
        self.vector_dir = Path(vector_dir)
        self.llm_model = model or os.getenv("LLM_MODEL", "mistral-small-latest")
        self.embed_model = None
        self.index = None
        self.documents = None
        self.metadatas = None
        self.client = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all required components."""
        print("🔧 Initializing RAG System...\n")
        
        # 1. Load embedding model
        print("  [1/4] Loading embedding model...", end=" ")
        try:
            self.embed_model = SentenceTransformer(EMBED_MODEL)
            print("✓")
        except Exception as e:
            print(f"✗\n❌ Failed to load embedding model: {e}")
            sys.exit(1)
        
        # 2. Load FAISS index
        print("  [2/4] Loading FAISS index...", end=" ")
        try:
            index_path = self.vector_dir / "faiss.index"
            if not index_path.exists():
                raise FileNotFoundError(f"Index not found at {index_path}")
            self.index = faiss.read_index(str(index_path))
            print(f"✓ ({self.index.ntotal:,} vectors)")
        except Exception as e:
            print(f"✗\n❌ Failed to load index: {e}")
            sys.exit(1)
        
        # 3. Load documents
        print("  [3/4] Loading documents...", end=" ")
        try:
            docs_path = self.vector_dir / "documents.pkl"
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"✓ ({len(self.documents):,} chunks)")
        except Exception as e:
            print(f"✗\n❌ Failed to load documents: {e}")
            sys.exit(1)
        
        # 4. Load metadata
        print("  [4/4] Loading metadata...", end=" ")
        try:
            meta_path = self.vector_dir / "metadata.json"
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
            print(f"✓ ({len(self.metadatas):,} entries)")
        except Exception as e:
            print(f"✗\n❌ Failed to load metadata: {e}")
            sys.exit(1)
        
        # 5. Initialize Mistral LLM
        print(f"\n  [5/5] Initializing Mistral ({self.llm_model})...", end=" ")
        try:
            if not MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY not set in .env")
            self.client = OpenAI(
                api_key=MISTRAL_API_KEY,
                base_url=MISTRAL_BASE_URL
            )
            print("✓\n")
        except Exception as e:
            print(f"✗\n❌ Failed to initialize LLM: {e}")
            sys.exit(1)
        
        print("✅ RAG System ready!\n")
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        # Encode query
        query_vector = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        
        # Search
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(distance)
                })
        
        return results
    
    def build_context(self, results: List[Dict[str, Any]], limit: int = CONTEXT_LIMIT) -> str:
        """Build context string from retrieved results."""
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(results, 1):
            text = result["text"]
            meta = result["metadata"]
            score = result["score"]
            
            # Format chunk with metadata
            chunk_header = f"[Source {i} | Score: {score:.3f} | Doc: {meta.get('doc_id', 'N/A')[:12]}...]"
            chunk_text = f"{chunk_header}\n{text}\n"
            
            # Check limit
            if total_chars + len(chunk_text) > limit:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n---\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""You are an AI assistant answering questions based on document analysis.

Context from relevant documents:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- Be specific and cite information from the sources when possible
- If the context doesn't contain enough information, say so
- Be concise but comprehensive

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, question: str, top_k: int = TOP_K, verbose: bool = True) -> Dict[str, Any]:
        """Complete RAG query pipeline."""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Query: {question}")
            print(f"{'='*70}\n")
        
        # 1. Retrieve
        if verbose:
            print("🔍 Retrieving relevant chunks...")
        results = self.retrieve(question, top_k=top_k)
        
        if not results:
            return {
                "query": question,
                "answer": "No relevant information found in the document store.",
                "sources": [],
                "context": ""
            }
        
        if verbose:
            print(f"   ✓ Found {len(results)} relevant chunks\n")
        
        # 2. Build context
        context = self.build_context(results)
        
        if verbose:
            print(f"📝 Context length: {len(context):,} characters\n")
        
        # 3. Generate answer
        if verbose:
            print("🤖 Generating answer...\n")
        answer = self.generate_answer(question, context)
        
        return {
            "query": question,
            "answer": answer,
            "sources": results,
            "context": context
        }


# -------------------------------------
# INTERACTIVE MODE
# -------------------------------------

def interactive_mode():
    """Run RAG system in interactive mode."""
    print("\n" + "="*70)
    print("RAG QUERY SYSTEM - INTERACTIVE MODE".center(70))
    print("="*70)
    
    # Initialize system
    rag = RAGSystem()
    
    print("Type your questions (or 'quit' to exit, 'help' for commands)\n")
    
    while True:
        try:
            # Get user input
            question = input("❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if question.lower() == 'help':
                print("\nCommands:")
                print("  - Type any question to get an answer")
                print("  - 'quit' or 'exit' to quit")
                print("  - 'help' to show this message\n")
                continue
            
            # Query
            result = rag.query(question, verbose=True)
            
            # Display answer
            print(f"💡 Answer:\n")
            print(result["answer"])
            print(f"\n📚 Sources used: {len(result['sources'])}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# -------------------------------------
# PROGRAMMATIC USAGE
# -------------------------------------

def simple_query(question: str, top_k: int = TOP_K) -> str:
    """Simple function for programmatic queries."""
    rag = RAGSystem()
    result = rag.query(question, top_k=top_k, verbose=False)
    return result["answer"]


# -------------------------------------
# MAIN
# -------------------------------------

if __name__ == "__main__":
    # Check if question provided as argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n{'='*70}")
        print("RAG QUERY SYSTEM".center(70))
        print(f"{'='*70}\n")
        
        rag = RAGSystem()
        result = rag.query(question, verbose=True)
        
        print(f"💡 Answer:\n")
        print(result["answer"])
        print(f"\n📚 Sources used: {len(result['sources'])}\n")
    else:
        # Run interactive mode
        interactive_mode()