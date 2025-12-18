# rag_library/pipeline/rag.py
# ENHANCED VERSION - Multiple models for different tasks

from __future__ import annotations

from typing import List, Optional, Any, Iterable, Dict
import random

from ..core.Models import Document, Chunk, QueryWithResults, ASRSegment
from ..chuncking import SimpleTextSplitter
from ..embeddings import BaseEmbedding
from ..vectordb import BaseVectorStore
from ..retriever import SimpleRetriever
from ..llm import BaseLLM


class RAGPipeline:
    """
    High-level RAG pipeline with multi-model support.
    
    You can now use different models for:
    - Question answering (main LLM)
    - Summarization (optional summary_llm)
    - Question generation (optional generation_llm)
    
    Original features:
      - add_documents()
      - add_segments()
      - answer_question()
    
    NEW features:
      - summarize_content()          # With optional dedicated model
      - answer_multiple_questions()  # Batch Q&A
      - generate_questions()         # With optional dedicated model
      - set_summary_llm()           # NEW: Set dedicated summary model
      - set_generation_llm()        # NEW: Set dedicated generation model
    """

    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        chunker: Optional[Any] = None,
        default_top_k: int = 5,
        summary_llm: Optional[BaseLLM] = None,      # NEW: Optional separate model
        generation_llm: Optional[BaseLLM] = None,   # NEW: Optional separate model
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm  # Main LLM for Q&A
        self.chunker = chunker or SimpleTextSplitter(max_chars=800, overlap=100)
        self.default_top_k = default_top_k

        # NEW: Optional specialized models
        self.summary_llm = summary_llm      # Falls back to main LLM if None
        self.generation_llm = generation_llm  # Falls back to main LLM if None

        self.retriever = SimpleRetriever(embedder=self.embedder, vector_store=self.vector_store)

    # ---------- NEW: Model Management ----------

    def set_summary_llm(self, llm: BaseLLM) -> None:
        """
        Set a dedicated model for summarization tasks.
        
        Args:
            llm: Language model to use for summaries
        """
        self.summary_llm = llm
    
    def set_generation_llm(self, llm: BaseLLM) -> None:
        """
        Set a dedicated model for question generation.
        
        Args:
            llm: Language model to use for generating questions
        """
        self.generation_llm = llm
    
    def get_active_models(self) -> Dict[str, str]:
        """
        Get info about which models are being used for each task.
        
        Returns:
            Dictionary with model info for each task
        """
        return {
            "qa_model": self.llm.__class__.__name__ if hasattr(self.llm, '__class__') else str(self.llm),
            "summary_model": (
                self.summary_llm.__class__.__name__ 
                if self.summary_llm and hasattr(self.summary_llm, '__class__') 
                else "Using main LLM"
            ),
            "generation_model": (
                self.generation_llm.__class__.__name__ 
                if self.generation_llm and hasattr(self.generation_llm, '__class__') 
                else "Using main LLM"
            ),
        }

    # ---------- Indexing (Original) ----------

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Chunk -> embed -> store."""
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

    def add_segments(self, segments: Iterable[ASRSegment], document_id: str) -> None:
        """Chunk -> embed -> store."""
        chunks = self.chunker.chunk_segments(segments=segments, document_id=document_id)

        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)

        for c, emb in zip(chunks, embeddings):
            c.embedding = emb

        self.vector_store.add_chunks(chunks)

    # ---------- Question Answering (Original) ----------

    def _build_prompt(self, question: str, query_with_results: QueryWithResults) -> str:
        """Build a simple RAG prompt from retrieved chunks."""
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
        return prompt

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ):
        """
        End-to-end RAG using main LLM.
        """
        if top_k is None:
            top_k = self.default_top_k

        query_results = self.retriever.retrieve(question, k=top_k)
        prompt = self._build_prompt(question, query_results)
        answer = self.llm.generate(prompt)  # Uses main LLM

        if not return_sources:
            return answer

        sources = [
            {
                "chunk": r.chunk,
                "chunk_id": r.chunk.id,
                "document_id": r.chunk.document_id,
                "score": r.score,
                "metadata": r.chunk.metadata,
            }
            for r in query_results.results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    # ---------- NEW FEATURE 1: Summarization (with optional model) ----------

    def summarize_content(
        self,
        max_chunks: int = 20,
        summary_type: str = "comprehensive",
        max_tokens: int = 1024,
        use_dedicated_model: bool = True,  # NEW: Option to use dedicated model
    ) -> str:
        """
        Summarize all indexed content.
        
        Args:
            max_chunks: Maximum chunks to use (samples if more exist)
            summary_type: "comprehensive", "brief", "keypoints", or "topics"
            max_tokens: Maximum tokens for the summary
            use_dedicated_model: If True and summary_llm is set, use it instead of main LLM
            
        Returns:
            Generated summary text
        """
        # Get all chunks from vector store
        all_chunks = self.vector_store._chunks
        
        if len(all_chunks) == 0:
            return "No content has been indexed yet. Please add documents or videos first."
        
        # Sample chunks if too many
        if len(all_chunks) > max_chunks:
            sampled_chunks = random.sample(all_chunks, max_chunks)
        else:
            sampled_chunks = all_chunks
        
        # Build context from chunks
        context_parts = []
        for chunk in sampled_chunks:
            context_parts.append(chunk.text)
        
        context = "\n\n".join(context_parts)
        
        # Build prompt based on summary type
        if summary_type == "brief":
            prompt = f"""Provide a brief 2-3 sentence summary of the following content:

{context}

Brief Summary:"""
        
        elif summary_type == "keypoints":
            prompt = f"""Extract the key points from the following content as a bullet list:

{context}

Key Points:
-"""
        
        elif summary_type == "topics":
            prompt = f"""Identify the main topics discussed in the following content:

{context}

Main Topics:
1."""
        
        else:  # comprehensive
            prompt = f"""Provide a comprehensive summary of the following content, covering all main ideas and important details:

{context}

Comprehensive Summary:"""
        
        # NEW: Choose which model to use
        llm_to_use = self.summary_llm if (use_dedicated_model and self.summary_llm) else self.llm
        
        # Generate summary
        summary = llm_to_use.generate(prompt, max_tokens=max_tokens)
        
        return summary.strip()

    # ---------- NEW FEATURE 2: Multiple Questions (uses main LLM) ----------

    def answer_multiple_questions(
        self,
        questions: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch using main LLM.
        """
        results = []
        
        for question in questions:
            try:
                answer_result = self.answer_question(
                    question,
                    top_k=top_k,
                    return_sources=True
                )
                
                results.append({
                    "question": question,
                    "answer": answer_result["answer"],
                    "sources": answer_result.get("sources", [])
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "answer": f"Error answering question: {str(e)}",
                    "sources": []
                })
        
        return results

    # ---------- NEW FEATURE 3: Generate Questions (with optional model) ----------

    def generate_questions(
        self,
        num_questions: int = 5,
        question_type: str = "comprehension",
        use_dedicated_model: bool = True,  # NEW: Option to use dedicated model
    ) -> List[str]:
        """
        Generate questions based on indexed content.
        
        Args:
            num_questions: Number of questions to generate
            question_type: "comprehension", "factual", or "analytical"
            use_dedicated_model: If True and generation_llm is set, use it instead of main LLM
            
        Returns:
            List of generated question strings
        """
        # Get sample of chunks
        all_chunks = self.vector_store._chunks
        
        if len(all_chunks) == 0:
            return []
        
        # Sample chunks for question generation
        sample_size = min(10, len(all_chunks))
        sampled_chunks = random.sample(all_chunks, sample_size)
        
        context = "\n\n".join([c.text for c in sampled_chunks])
        
        # Build prompt based on question type
        if question_type == "factual":
            prompt = f"""Based on the following content, generate {num_questions} factual questions that can be answered directly from the text. Each question should test specific facts or details mentioned in the content.

Content:
{context}

Generate exactly {num_questions} factual questions (one per line):
1."""
        
        elif question_type == "analytical":
            prompt = f"""Based on the following content, generate {num_questions} analytical questions that require deeper understanding and critical thinking.

Content:
{context}

Generate exactly {num_questions} analytical questions (one per line):
1."""
        
        else:  # comprehension
            prompt = f"""Based on the following content, generate {num_questions} comprehension questions that test understanding of the main ideas.

Content:
{context}

Generate exactly {num_questions} comprehension questions (one per line):
1."""
        
        # NEW: Choose which model to use
        llm_to_use = self.generation_llm if (use_dedicated_model and self.generation_llm) else self.llm
        
        # Generate questions
        generated = llm_to_use.generate(prompt, max_tokens=512)
        
        # Parse questions from generated text
        questions = []
        for line in generated.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                question = line.lstrip('0123456789.-•) ').strip()
                if question and '?' in question:
                    questions.append(question)
                    if len(questions) >= num_questions:
                        break
        
        return questions

    # ---------- Helper Methods ----------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed content."""
        all_chunks = self.vector_store._chunks
        
        unique_docs = set(chunk.document_id for chunk in all_chunks)
        video_chunks = sum(1 for chunk in all_chunks if "timestamp" in chunk.metadata)
        total_chars = sum(len(chunk.text) for chunk in all_chunks)
        
        return {
            "total_chunks": len(all_chunks),
            "unique_documents": len(unique_docs),
            "video_chunks": video_chunks,
            "document_chunks": len(all_chunks) - video_chunks,
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(all_chunks) if all_chunks else 0,
        }