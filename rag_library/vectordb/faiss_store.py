from __future__ import annotations

from typing import List

import numpy as np

from ..core.Models import Chunk, RetrievalResult
from .base import BaseVectorStore

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError(
        "faiss is required for FaissVectorStore.\n"
        "Install it with: pip install faiss-cpu"
    ) from e


class FaissVectorStore(BaseVectorStore):
    """
    FAISS-based vector store.

    Assumes embeddings are L2-normalized if you use IndexFlatIP
    (inner product as cosine similarity).
    """

    def __init__(self, dim: int):
        self.dim = dim
        # Using inner product index (works as cosine if embeddings normalized)
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        # stack embeddings
        embs = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id} has no embedding.")
            if c.embedding.shape[-1] != self.dim:
                raise ValueError(
                    f"Chunk {c.id} embedding dim={c.embedding.shape[-1]} "
                    f"does not match index dim={self.dim}"
                )
            embs.append(c.embedding)

        if not embs:
            return

        embs = np.vstack(embs).astype(np.float32)
    #    add normalization for INNER PRODUCT
    # embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        self.index.add(embs)
        self._chunks.extend(chunks)

    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        if len(self._chunks) == 0:
            return []

        q = query_embedding.astype(np.float32).reshape(1, -1)
        if q.shape[-1] != self.dim:
            raise ValueError(
                f"Query embedding dim={q.shape[-1]} does not match index dim={self.dim}"
            )

        k = min(k, len(self._chunks))
        distances, indices = self.index.search(q, k)  # shapes: (1,k), (1,k)

        d = distances[0]
        idxs = indices[0]

        results: List[RetrievalResult] = []
        for score, idx in zip(d, idxs):
            if idx < 0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[int(idx)],
                    score=float(score),  # inner product similarity
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)
