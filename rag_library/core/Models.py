from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # <--- NEW



@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class QueryWithResults:
    query: str
    results: List[RetrievalResult]

@dataclass
class ASRSegment:
    """
    Represents a single ASR segment from a video/audio transcript.

    Example:
        start: 1.2
        end:   4.7
        text: "The heart pumps blood to the lungs..."
    """
    start: float
    end: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


