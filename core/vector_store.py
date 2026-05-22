"""
Vector Store for GrantMind AI.

What is a Vector Database?
--------------------------
A vector database is a specialized storage system designed to store and query high-dimensional 
vectors (embeddings) efficiently. Standard relational databases (like SQLite or PostgreSQL) 
index data using rows and columns and search via exact matches. In contrast, a vector database:
1. Indexes dense vector representations of unstructured text.
2. Uses approximate nearest neighbor (ANN) search algorithms (like HNSW) to quickly locate 
   vectors closest to a given query vector.
3. Calculates distances (like cosine similarity or L2 distance) to assess similarity.

In our system, ChromaDB persists text chunks and their embeddings locally in the `chroma_db` folder. 
This acts as our "external memory," letting us find precise context within large documents in milliseconds.
"""

import os
import chromadb
from typing import List, Dict, Any

# Resolve persistent path relative to this file to prevent random folder creation
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
PERSIST_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Creates and returns a local persistent ChromaDB client.

    Returns:
        chromadb.PersistentClient: The persistent client instance.
    """
    return chromadb.PersistentClient(path=PERSIST_PATH)

def get_collection() -> chromadb.Collection:
    """
    Gets or creates the 'grant_documents' collection in ChromaDB.

    Returns:
        chromadb.Collection: The document collection instance.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name="grant_documents")

def insert_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """
    Inserts chunk texts, embeddings, and metadata into ChromaDB.

    Parameters:
        chunks (List[Dict[str, Any]]): List of chunk objects with text & metadata.
        embeddings (List[List[float]]): Corresponding dense vector embeddings.
    """
    if not chunks or not embeddings:
        return
    collection = get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for idx, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        # Generate an absolute unique ID per chunk
        chunk_id = f"{meta['filename']}_p{meta['page_number']}_c{meta['chunk_index']}_{idx}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(meta)
        
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def search_similar(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the collection for the top K most similar text chunks.

    Parameters:
        query_embedding (List[float]): Dense vector embedding of the query.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict[str, Any]]: List of matching chunks with text and metadata.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    formatted_results = []
    if not results or not results["documents"] or not results["documents"][0]:
        return formatted_results
        
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    for idx, doc in enumerate(docs):
        formatted_results.append({
            "text": doc,
            "metadata": metadatas[idx]
        })
        
    return formatted_results

def clear_collection() -> None:
    """
    Drops and recreates the 'grant_documents' collection, resetting the database.
    """
    client = get_chroma_client()
    try:
        client.delete_collection("grant_documents")
    except Exception:
        # Collection might not exist yet, which is safe to ignore
        pass
    # Re-initialize collection
    client.get_or_create_collection(name="grant_documents")
