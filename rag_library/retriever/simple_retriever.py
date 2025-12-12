from __future__ import annotations

from typing import List

from ..core.Models import QueryWithResults, RetrievalResult
from ..embeddings import BaseEmbedding
from ..vectordb import BaseVectorStore


class SimpleRetriever:

    def __init__(self, embedder: BaseEmbedding, vector_store: BaseVectorStore, default_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_k = default_k

    def retrieve(self, query: str, k: int | None = None) -> QueryWithResults:
        if k is None:
            k = self.default_k

        # 1) embed query
        query_emb = self.embedder.encode_one(query)

        # 2) similarity search
        results: List[RetrievalResult] = self.vector_store.similarity_search_by_vector(
            query_emb, k=k
        )

        # 3) wrap into QueryWithResults for debugging
        return QueryWithResults(query=query, results=results)
