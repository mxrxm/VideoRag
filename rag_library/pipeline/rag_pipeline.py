from __future__ import annotations

from typing import List, Optional, Any, Iterable

from ..core.Models import Document, Chunk, QueryWithResults,ASRSegment
from ..chuncking import SimpleTextSplitter
from ..embeddings import BaseEmbedding
from ..vectordb import BaseVectorStore
from ..retriever import SimpleRetriever
from ..llm import BaseLLM


class RAGPipeline:
    """
    High-level RAG pipeline:

      - add_documents()
      - answer_question()

    It wires together:
      - chunker (e.g. SimpleTextSplitter)
      - embedder (HuggingFaceEmbedding)
      - vector store (InMemory, FAISS, Chroma)
      - retriever (SimpleRetriever)
      - LLM (HuggingFaceLLM)
    """

    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        chunker: Optional[Any] = None,
        default_top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.chunker = chunker or SimpleTextSplitter(max_chars=800, overlap=100)
        self.default_top_k = default_top_k

        self.retriever = SimpleRetriever(embedder=self.embedder, vector_store=self.vector_store)

    # ---------- Indexing ----------

    def add_documents(self, documents: Iterable[Document]) -> None:
        """
        Chunk -> embed -> store.
        """
        all_chunks: List[Chunk] = []

        for doc in documents:
            chunks = self.chunker.split_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.encode(texts)

        for c, emb in zip(all_chunks, embeddings):
            c.embedding = emb

        self.vector_store.add_chunks(all_chunks)


    def add_segments(self, segments: Iterable[ASRSegment],document_id :str) -> None:
        """
        Chunk -> embed -> store.
        """
        chunks = self.chunker.chunk_segments(segments=segments,document_id=document_id)

        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)

        for c, emb in zip(chunks, embeddings):
            c.embedding = emb

        self.vector_store.add_chunks(chunks)


    def _build_prompt(self, question: str, query_with_results: QueryWithResults) -> str:
        """
        Build a simple RAG prompt from retrieved chunks.
        You can customize this template later.
        """
        context_parts = []
        for idx, r in enumerate(query_with_results.results):
            context_parts.append(f"[CHUNK {idx}]\n{r.chunk.text}\n")

        context_str = "\n".join(context_parts)

        prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer the question.\n\n"
            "Context:\n"
            f"{context_str}\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer clearly and concisely:\n"
        )
        # print(context_str)
        return prompt

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ):
        """
        End-to-end RAG:
          1) retrieve relevant chunks
          2) build prompt
          3) generate answer with LLM

        If return_sources=True, also returns metadata of retrieved chunks.
        """
        if top_k is None:
            top_k = self.default_top_k

        query_results = self.retriever.retrieve(question, k=top_k)
        prompt = self._build_prompt(question, query_results)
        answer = self.llm.generate(prompt)

        if not return_sources:
            return answer

        sources = [
            {
                "chunk_id": r.chunk.id,
                "document_id": r.chunk.document_id,
                "score": r.score,
                "metadata": r.chunk.metadata,
            }
            for r in query_results.results
        ]

        return {
            "answer": answer
            # "sources": sources,
        }
