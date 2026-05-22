"""
Embedding Engine for GrantMind AI.

What are Embeddings and Why Do We Use Them?
-------------------------------------------
An embedding is a vector (a list of real numbers) that represents the semantic meaning 
of a piece of text (a word, a sentence, or a paragraph). Unlike simple word matching, 
which searches for exact spelling, embeddings map text into a continuous high-dimensional 
vector space (384 dimensions for all-MiniLM-L6-v2). 

In this semantic space:
- Text snippets with similar meanings are positioned close to one another (high cosine similarity).
- Syntactically different but semantically identical phrases (e.g., "funding requested" and 
  "budget requirements") align closely.
- This enables semantic search: finding the most relevant context even if the user 
  doesn't use the exact keywords in the query.
"""

from typing import List
from sentence_transformers import SentenceTransformer

# Global model cache to avoid reloading the model on every function call
_MODEL_CACHE = None

def get_embedding_model() -> SentenceTransformer:
    """
    Retrieves or loads the cached local SentenceTransformer model.

    Returns:
        SentenceTransformer: The loaded all-MiniLM-L6-v2 embedding model.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        # Using a fast, lightweight, and local model (384 dimensions, ~90MB)
        _MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL_CACHE

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of text strings.

    Parameters:
        texts (List[str]): List of clean text chunks to embed.

    Returns:
        List[List[float]]: A list of 384-dimensional dense vectors.
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [vec.tolist() for vec in embeddings]

def embed_query(query: str) -> List[float]:
    """
    Generates a vector embedding for a single user query.

    Parameters:
        query (str): The search query or question string.

    Returns:
        List[float]: A single 384-dimensional dense vector.
    """
    model = get_embedding_model()
    # encode accepts a string and returns a single numpy array
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()
