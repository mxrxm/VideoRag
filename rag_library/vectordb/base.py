from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..core.Models import Chunk, RetrievalResult


class BaseVectorStore(ABC):
    """
    Abstract base class for all vector stores (FAISS, Chroma, in-memory, etc.).
    Stores Chunk objects + their embeddings and supports similarity search.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks (with embeddings already attached) to the store.
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Search most similar chunks for a given query embedding.
        Returns list of RetrievalResult (chunk + score).
        """
        raise NotImplementedError

    def __len__(self) -> int:  # optional but convenient
        return 0


class InMemoryVectorStore(BaseVectorStore):
    """
    Simple in-memory vector store using numpy.
    Great for testing before using FAISS/Chroma.
    """

    def __init__(self):
        self._embeddings: np.ndarray | None = None  # shape (n, dim)
        self._chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        # collect embeddings
        new_embs = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id} has no embedding.")
            new_embs.append(c.embedding)

        new_embs = np.vstack(new_embs).astype(np.float32)

        if self._embeddings is None:
            self._embeddings = new_embs
        else:
            self._embeddings = np.vstack([self._embeddings, new_embs])

        self._chunks.extend(chunks)

    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        docs = self._embeddings  # (n, dim)

        # cosine similarity
        dot = np.dot(docs, query.T).reshape(-1)  # (n,)
        norms = np.linalg.norm(docs, axis=1) * (np.linalg.norm(query) + 1e-12)
        sims = dot / (norms + 1e-12)

        # top-k indices
        k = min(k, len(self._chunks))
        top_idx = np.argsort(sims)[::-1][:k]

        results: List[RetrievalResult] = []
        for idx in top_idx:
            results.append(
                RetrievalResult(
                    chunk=self._chunks[int(idx)],
                    score=float(sims[int(idx)]),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)
