# RAG/vector_store_builder.py
# Simple vector store builder using FAISS - Master Merger Only

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -------------------------------------
# CONFIG
# -------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "agent_system")

VECTOR_DIR = "./vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 600
MIN_CHUNK = 100


# -------------------------------------
# TEXT EXTRACTION
# -------------------------------------

def extract_master_text(doc: Dict[str, Any]) -> str:
    """Extract all relevant text from master_merger_results."""
    fields = []
    
    # 1. Executive summary (most important)
    exec_sum = doc.get("executive_summary", "")
    if isinstance(exec_sum, str) and exec_sum.strip():
        fields.append(exec_sum.strip())
    
    # 2. Detailed synthesis sections
    detailed = doc.get("detailed_synthesis", {})
    if isinstance(detailed, dict):
        # Section syntheses
        sections = detailed.get("sections", [])
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    synthesis = section.get("synthesis", "")
                    if isinstance(synthesis, str) and synthesis.strip():
                        fields.append(synthesis.strip())
        
        # Cross-section analysis
        cross_analysis = detailed.get("cross_section_analysis", "")
        if isinstance(cross_analysis, str) and cross_analysis.strip():
            fields.append(cross_analysis.strip())
        
        # Technical deep dive
        tech_dive = detailed.get("technical_deep_dive", "")
        if isinstance(tech_dive, str) and tech_dive.strip():
            fields.append(tech_dive.strip())
    
    # 3. Insights and conclusions
    insights = doc.get("insights_and_conclusions", {})
    if isinstance(insights, dict):
        # Key findings
        findings = insights.get("key_findings", [])
        if isinstance(findings, list) and findings:
            findings_text = "\n".join([str(f) for f in findings if f])
            if findings_text:
                fields.append("Key Findings:\n" + findings_text)
        
        # Conclusions
        conclusions = insights.get("conclusions", "")
        if isinstance(conclusions, str) and conclusions.strip():
            fields.append("Conclusions:\n" + conclusions.strip())
        
        # Implications
        implications = insights.get("implications", [])
        if isinstance(implications, list) and implications:
            impl_text = "\n".join([str(i) for i in implications if i])
            if impl_text:
                fields.append("Implications:\n" + impl_text)
    
    return "\n\n".join(fields)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove problematic characters
    text = text.replace('\x00', ' ')
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    # Normalize whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks."""
    text = clean_text(text)
    
    if not text or len(text) < MIN_CHUNK:
        return [text] if text else []
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        
        if current_length + word_len > size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK:
                chunks.append(chunk_text)
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK:
            chunks.append(chunk_text)
    
    return chunks if chunks else ([text] if len(text) >= MIN_CHUNK else [])


# -------------------------------------
# METADATA
# -------------------------------------

def get_metadata(doc: Dict[str, Any]) -> Dict[str, str]:
    """Extract metadata from master merger document."""
    meta = {
        "source": "master_merger",
        "doc_id": str(doc.get("_id", ""))
    }
    
    # Add relevant IDs
    if "agent_id" in doc:
        meta["agent_id"] = str(doc["agent_id"])
    
    # Add document themes if available
    metadata_obj = doc.get("metadata", {})
    if isinstance(metadata_obj, dict):
        themes = metadata_obj.get("document_themes", [])
        if isinstance(themes, list) and themes:
            meta["themes"] = ", ".join([str(t) for t in themes[:3]])  # First 3 themes
    
    return meta


# -------------------------------------
# MAIN BUILDER
# -------------------------------------

def build_vector_store():
    print("\n" + "="*70)
    print("BUILDING VECTOR STORE (MASTER MERGER ONLY)".center(70))
    print("="*70 + "\n")

    if not MONGO_URI:
        print("❌ MONGO_URI not set")
        sys.exit(1)

    # Connect MongoDB
    print("[1/5] Connecting to MongoDB...")
    try:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo[MONGO_DB]
        db.list_collection_names()
        print("      ✅ Connected\n")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        sys.exit(1)

    # Load model
    print("[2/5] Loading embedding model...")
    try:
        model = SentenceTransformer(EMBED_MODEL)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"      ✅ Loaded {EMBED_MODEL} ({embedding_dim}D)\n")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        sys.exit(1)

    # Initialize storage
    print("[3/5] Initializing FAISS index...")
    try:
        Path(VECTOR_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create FAISS index (cosine similarity)
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Storage for documents and metadata
        documents = []
        metadatas = []
        
        print("      ✅ Ready\n")
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        sys.exit(1)

    # Process master_merger_results only
    print("[4/5] Processing master merger documents...\n")
    
    all_embeddings = []
    
    try:
        coll = db["master_merger_results"]
        docs = list(coll.find({}))
        
        if not docs:
            print("     ❌ No documents found in master_merger_results!")
            sys.exit(1)
        
        print(f"  📁 Found {len(docs)} master documents\n")
        
        for doc_num, doc in enumerate(docs, 1):
            try:
                doc_id = str(doc.get("_id", "unknown"))
                agent_id = doc.get("agent_id", "N/A")
                
                print(f"  [{doc_num}/{len(docs)}] {agent_id[:20]}...")
                print(f"         Doc ID: {doc_id[:16]}...")
                
                # Extract text
                text = extract_master_text(doc)
                
                if not text:
                    print(f"         ⚠️  No text extracted")
                    print(f"         Available keys: {list(doc.keys())[:5]}")
                    continue
                
                print(f"         ✓ Extracted {len(text):,} chars")
                
                # Chunk
                chunks = chunk_text(text)
                if not chunks:
                    print(f"         ⚠️  No chunks created")
                    continue
                
                print(f"         ✓ Created {len(chunks)} chunks")
                
                # Get metadata
                base_meta = get_metadata(doc)
                
                # Encode
                print(f"         → Encoding...", end=" ")
                embeddings = model.encode(
                    chunks,
                    show_progress_bar=False,
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                print("✓")
                
                # Store
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    meta = base_meta.copy()
                    meta["chunk_idx"] = str(i)
                    meta["total_chunks"] = str(len(chunks))
                    
                    documents.append(chunk)
                    metadatas.append(meta)
                    all_embeddings.append(embedding)
                
                print(f"         ✅ Added {len(chunks)} chunks to index\n")
                
            except Exception as e:
                print(f"         ❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        
    except Exception as e:
        print(f"     ❌ Collection error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Build FAISS index
    print("[5/5] Building FAISS index...")
    try:
        if not all_embeddings:
            print("      ❌ No embeddings to index!")
            sys.exit(1)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        print(f"      → Adding {len(embeddings_array):,} vectors to index...")
        
        # Add to index
        index.add(embeddings_array)
        
        # Save index
        index_path = Path(VECTOR_DIR) / "faiss.index"
        faiss.write_index(index, str(index_path))
        print(f"      ✅ Saved index: {index_path}")
        
        # Save documents
        docs_path = Path(VECTOR_DIR) / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
        print(f"      ✅ Saved documents: {docs_path}")
        
        # Save metadata
        meta_path = Path(VECTOR_DIR) / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadatas, f, indent=2, ensure_ascii=False)
        print(f"      ✅ Saved metadata: {meta_path}")
        
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "="*70)
    print("COMPLETE".center(70))
    print("="*70)
    print(f"\nTotal chunks indexed: {len(documents):,}")
    print(f"Vector dimensions: {embedding_dim}")
    print(f"Storage location: {VECTOR_DIR}/")
    print(f"  • faiss.index ({index.ntotal:,} vectors)")
    print(f"  • documents.pkl")
    print(f"  • metadata.json\n")
    print("✅ Ready for RAG queries!\n")


if __name__ == "__main__":
    try:
        build_vector_store()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)