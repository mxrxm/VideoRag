from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np

from ..core.Models import Chunk, RetrievalResult
from .base import BaseVectorStore

try:
    import chromadb
except ImportError as e:
    raise ImportError(
        "chromadb is required for ChromaVectorStore.\n"
        "Install it with: pip install chromadb"
    ) from e


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma-based vector store.

    We use external embeddings (from your embedder) and pass them directly
    to Chroma. Distances are usually L2, so we convert them into a
    similarity-like score (negated distance).
    """

    def __init__(
        self,
        collection_name: str = "raglib_collection",
        client: Optional["chromadb.Client"] = None,
    ):
        self._client = client or chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self._id_to_chunk: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[List[float]] = []

        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id} has no embedding.")

            chunk_id = c.id
            ids.append(chunk_id)
            docs.append(c.text)
            metas.append(c.metadata)
            embs.append(c.embedding.astype(float).tolist())

            self._id_to_chunk[chunk_id] = c

        if not ids:
            return

        self._collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )

    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        if len(self._id_to_chunk) == 0:
            return []

        q = query_embedding.astype(float).tolist()

        results = self._collection.query(
            query_embeddings=[q],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        ids_list = results.get("ids", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        out: List[RetrievalResult] = []

        for chunk_id, dist in zip(ids_list, distances_list):
            chunk = self._id_to_chunk.get(chunk_id)
            if chunk is None:
                continue
            # smaller distance -> more similar; we convert to a negative distance score
            score = -float(dist)
            out.append(RetrievalResult(chunk=chunk, score=score))

        return out

    def __len__(self) -> int:
        return len(self._id_to_chunk)
