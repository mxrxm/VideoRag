from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable, TYPE_CHECKING

import numpy as np

from ..core.Models import Document, Chunk , ASRSegment

if TYPE_CHECKING:
    # For type hints only (avoids circular imports at runtime)
    from embeddings import BaseEmbedding


# -----------------------------
# Basic helpers
# -----------------------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Normalize whitespace
      - Remove extra linebreaks
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter using regex.

    For production you may switch to spaCy or nltk,
    but this works well as an MVP.
    """
    text = clean_text(text)
    if not text:
        return []
    # Split on . ! ? followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Remove empty pieces
    return [s.strip() for s in sentences if s.strip()]


# -----------------------------
# 1) SimpleTextSplitter
# -----------------------------

class SimpleTextSplitter:
    """
    A simple character-based text splitter with overlap.
    Equivalent to LangChain's CharacterTextSplitter (roughly).
    """

    def __init__(self, max_chars: int = 800, overlap: int = 100):
        if overlap >= max_chars:
            raise ValueError("overlap must be smaller than max_chars")

        self.max_chars = max_chars
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        text = clean_text(text)
        chunks: List[str] = []

        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chars, length)
            part = text[start:end].strip()
            if part:
                chunks.append(part)

            if end == length:
                break

            start += self.max_chars - self.overlap

        return chunks

    def split_document(self, document: Document) -> List[Chunk]:
        raw_chunks = self.split_text(document.text)
        chunks: List[Chunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{document.id}_chunk_{i}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata={**document.metadata, "chunk_index": i},
                    embedding=None,
                )
            )
        return chunks


def split_document_into_chunks(
    document: Document,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Convenience function using SimpleTextSplitter.
    """
    splitter = SimpleTextSplitter(max_chars=max_chars, overlap=overlap)
    return splitter.split_document(document)


# -----------------------------
# 2) SentenceTextSplitter
# -----------------------------

class SentenceTextSplitter:
    """
    Groups sentences into chunks.

    Good for research PDFs where you want to keep sentence boundaries.
    """

    def __init__(
        self,
        max_sentences: int = 5,
        overlap_sentences: int = 1,
        max_chars: int = 2000,
    ):
        if overlap_sentences >= max_sentences:
            raise ValueError("overlap_sentences must be < max_sentences")

        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences
        self.max_chars = max_chars

    def split_text(self, text: str) -> List[str]:
        sentences = split_sentences(text)
        chunks: List[str] = []

        i = 0
        n = len(sentences)

        while i < n:
            current_sentences: List[str] = []
            total_chars = 0
            count = 0

            while i < n and count < self.max_sentences:
                s = sentences[i]
                if total_chars + len(s) > self.max_chars and count > 0:
                    break
                current_sentences.append(s)
                total_chars += len(s) + 1
                count += 1
                i += 1

            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())

            if i >= n:
                break

            i = max(0, i - self.overlap_sentences)

        return chunks

    def split_document(self, document: Document) -> List[Chunk]:
        raw_chunks = self.split_text(document.text)
        chunks: List[Chunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{document.id}_sent_chunk_{i}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata={**document.metadata, "chunk_index": i},
                    embedding=None,
                )
            )
        return chunks


# -----------------------------
# 3) SemanticChunker (embedding-based)
# -----------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class SemanticChunker:
    """
    Embedding-based semantic chunker.

    Pipeline:
      1) Split into sentences
      2) Embed each sentence
      3) Group sentences into chunks as long as they stay semantically similar
         (cosine similarity between consecutive sentences >= threshold)
    """

    def __init__(
        self,
        embedder: "BaseEmbedding",
        similarity_threshold: float = 0.65,
        max_sentences_per_chunk: int = 8,
        min_sentences_per_chunk: int = 1,
        max_chars_per_chunk: int = 2000,  # NEW: Prevent huge chunks
    ):
        if min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be > 0")
        if max_sentences_per_chunk < min_sentences_per_chunk:
            raise ValueError("max_sentences_per_chunk must be >= min_sentences_per_chunk")

        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk  # NEW

    def split_text(self, text: str) -> List[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        sent_embeddings = self.embedder.encode(sentences)
        chunks: List[str] = []

        current_chunk_sents: List[str] = [sentences[0]]
        current_char_count = len(sentences[0])  # NEW: Track character count

        for i in range(1, len(sentences)):
            prev_emb = sent_embeddings[i - 1]
            cur_emb = sent_embeddings[i]
            sim = _cosine_similarity(prev_emb, cur_emb)

            chunk_len = len(current_chunk_sents)
            sentence_len = len(sentences[i])  # NEW

            should_break = False

            # NEW: Check character limit first
            if current_char_count + sentence_len > self.max_chars_per_chunk:
                should_break = True
            elif chunk_len >= self.max_sentences_per_chunk:
                should_break = True
            elif chunk_len >= self.min_sentences_per_chunk and sim < self.similarity_threshold:
                should_break = True

            if should_break:
                chunk_text = " ".join(current_chunk_sents).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk_sents = [sentences[i]]
                current_char_count = sentence_len  # NEW: Reset counter
            else:
                current_chunk_sents.append(sentences[i])
                current_char_count += sentence_len + 1  # NEW: +1 for space

        if current_chunk_sents:
            chunk_text = " ".join(current_chunk_sents).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def split_document(self, document: Document) -> List[Chunk]:
        raw_chunks = self.split_text(document.text)
        chunks: List[Chunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{document.id}_semantic_chunk_{i}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata={**document.metadata, "chunk_index": i},
                    embedding=None,
                )
            )
        return chunks

# -----------------------------
# 4) ASR / Video Timestamp Chunker
# -----------------------------



class ASRTimestampChunker:
    """
    Chunk video transcripts using timestamps.

    - Groups segments into windows of at most `max_seconds`.
    - Creates overlapping windows with `overlap_seconds`.

    This is ideal for RAG over video lectures.
    """

    def __init__(self, max_seconds: float = 15.0, overlap_seconds: float = 3.0):
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds must be >= 0")
        if overlap_seconds >= max_seconds:
            raise ValueError("overlap_seconds must be < max_seconds")

        self.max_seconds = max_seconds
        self.overlap_seconds = overlap_seconds

    def chunk_segments(
        self,
        segments: Iterable[ASRSegment],
        document_id: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        segments: list/iterable of ASRSegment (must have start, end, text)
        document_id: ID of the parent Document / video
        base_metadata: metadata applied to all chunks (e.g., {"source": "video1.mp4"})
        """
        base_metadata = base_metadata or {}

        segs = sorted(list(segments), key=lambda s: s.start)
        n = len(segs)
        chunks: List[Chunk] = []

        i = 0
        chunk_idx = 0

        while i < n:
            window_start = segs[i].start
            window_end = segs[i].end
            texts: List[str] = []
            seg_metadatas: List[Dict[str, Any]] = []
            j = i

            while j < n and (segs[j].end - window_start) <= self.max_seconds:
                texts.append(segs[j].text.strip())
                seg_metadatas.append(segs[j].metadata)
                window_end = segs[j].end
                j += 1

            chunk_text = clean_text(" ".join(texts))
            if chunk_text:
                merged_metadata: Dict[str, Any] = {
                    **base_metadata,
                    "start": window_start,
                    "end": window_end,
                    "chunk_index": chunk_idx,
                }
                if seg_metadatas:
                    merged_metadata["segments_metadata"] = seg_metadatas

                chunk = Chunk(
                    id=f"{document_id}_asr_chunk_{chunk_idx}",
                    document_id=document_id,
                    text=chunk_text,
                    metadata=merged_metadata,
                    embedding=None,
                )
                chunks.append(chunk)
                chunk_idx += 1

            if j >= n:
                break

            target_restart_time = window_end - self.overlap_seconds
            k = j
            while k > 0 and segs[k - 1].start >= target_restart_time:
                k -= 1
            i = max(k, j - 1)

        return chunks
