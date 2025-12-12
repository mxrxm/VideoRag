# embeddings/utils.py
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.Models import Chunk
    from .base import BaseEmbedding


def embed_chunks(
    chunks: List["Chunk"],
    embedder: "BaseEmbedding",
    show_progress: bool = True,
    skip_if_exists: bool = True,
) -> List["Chunk"]:
    """
    Embed a list of chunks (modifies in place).
    
    Args:
        chunks: List of Chunk objects
        embedder: Embedding model
        show_progress: Show progress bar
        skip_if_exists: Skip chunks that already have embeddings
        
    Returns:
        Same list of chunks (for chaining)
    """
    # Find chunks that need embedding
    chunks_to_embed = []
    indices_to_embed = []
    
    for i, chunk in enumerate(chunks):
        if skip_if_exists and chunk.embedding is not None:
            continue
        chunks_to_embed.append(chunk)
        indices_to_embed.append(i)
    
    if not chunks_to_embed:
        if show_progress:
            print("All chunks already have embeddings")
        return chunks
    
    # Extract texts
    texts = [chunk.text for chunk in chunks_to_embed]
    
    if show_progress:
        print(f"Embedding {len(texts)} chunks...")
    
    # Encode (use encode_batch if available for progress)
    if hasattr(embedder, 'encode_batch'):
        embeddings = embedder.encode_batch(texts, show_progress=show_progress)
    else:
        embeddings = embedder.encode(texts)
    
    # Assign back
    for chunk, embedding in zip(chunks_to_embed, embeddings):
        chunk.embedding = embedding
    
    return chunks


def find_most_similar(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find most similar embeddings to a query.
    
    Args:
        query_embedding: Query embedding (1D array)
        candidate_embeddings: Candidate embeddings (2D array)
        top_k: Number of results
        
    Returns:
        Tuple of (indices, similarities)
    """
    # Normalize query
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    
    # Normalize candidates
    candidates_norm = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-12
    )
    
    # Compute similarities
    similarities = np.dot(candidates_norm, query_norm)
    
    # Get top-k
    top_k = min(top_k, len(similarities))
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
    
    return top_indices, similarities[top_indices]