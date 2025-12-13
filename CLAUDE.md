# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoRAG is a video retrieval-augmented generation system that processes videos (local or URLs) to extract multi-modal content (audio transcription + on-screen text via OCR) and enables semantic Q&A over video content through a RAG pipeline.

**Important**: The `rag_library/` module is a general-purpose RAG framework that works with both video content (via ASRSegments) and traditional documents (PDF/PPTX via Document loaders). Use `test_chunkers.py` to experiment with the library on different content types.

## Setup and Running

### Environment Setup
```bash
# Windows - Initial setup with virtual environment
setup.bat

# Windows - Subsequent runs
run.bat

# Or use Conda-based launcher (legacy)
python Launcher.py
```

### Running the Application
```bash
# Streamlit Web Interface (Recommended for most users)
run_streamlit.bat
# Or: streamlit run streamlit_app.py

# Interactive CLI mode - Full video processing + RAG Q&A
python main.py

# Test chunking strategies on different document types (PDF/PPTX/Video JSON)
python rag_library/test_chunkers.py

# Direct mode examples (uncomment in main.py)
# process_single_video_direct("video.mp4")
# process_single_video_direct("https://youtube.com/watch?v=ID")
```

### Streamlit Web Interface Features

The Streamlit app (`streamlit_app.py`) provides a user-friendly web interface with:

**Configuration Options (Sidebar):**
- **Embedding Models**: BGE (Small/Base/Large), MiniLM, MPNet
- **LLMs**: Qwen (0.6B/1.8B), Phi-2, TinyLlama
- **Vector Databases**: FAISS, ChromaDB, In-Memory
- **Chunkers**: Simple, Sentence, Semantic, ASR Timestamp

**Main Features:**
- Upload and process PDF, PowerPoint, or Video files
- Real-time processing status and progress indicators
- Interactive Q&A chat interface
- Chat history tracking
- Visual feedback with chunk counts and indexed content

**Workflow:**
1. Configure models in sidebar (or use defaults)
2. Click "Initialize RAG Pipeline"
3. Upload your file (PDF/PPTX/Video)
4. Click "Process & Index"
5. Ask questions in the chat interface

### Key Configuration (CLI - main.py)
```python
WHISPER_MODEL = "base"      # tiny, base, small, medium, large
FPS = 1                     # Frames per second extraction rate
USE_OCR = True             # Enable on-screen text extraction
OCR_LANGUAGES = ['en']     # Languages for OCR
OCR_CONFIDENCE = 0.5       # Minimum OCR confidence (0-1)
DELETE_ORIGINAL = False    # Clean up downloaded videos
```

## Architecture

### Two-Stage Pipeline Design

**Stage 1: Video Processing** (`scripts/`)
```
Input Video (local/URL)
→ VideoDownloader (yt-dlp/HTTP)
→ AudioExtractor (FFmpeg) + FrameExtractor (FFmpeg)
→ AudioTranscriber (Whisper) + OCRExtractor (EasyOCR)
→ JSON output (transcripts/{name}_complete.json)
```

**Stage 2: RAG Pipeline** (`rag_library/`)
```
ASRSegments from JSON
→ ASRTimestampChunker (time-windowed chunks)
→ HuggingFaceEmbedding (dense vectors)
→ FaissVectorStore (indexed)
→ answer_question() → Retriever → LLM → Answer
```

### Core Data Models (`rag_library/core/Models.py`)

```python
@dataclass
class ASRSegment:
    start: float          # Timestamp in seconds
    end: float
    text: str            # Combined ASR + OCR text
    metadata: dict       # Contains 'ocr_text' if available

@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    embedding: np.ndarray  # Added during indexing
    metadata: dict         # Preserves timestamp info

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float          # Cosine similarity
```

### Video Processing Flow

**VideoProcessor** (`scripts/video_processor.py`) orchestrates:
1. Download if URL (supports YouTube, Vimeo, direct links via yt-dlp)
2. Extract audio → WAV via FFmpeg
3. Extract frames at configured FPS
4. Transcribe audio → segments with timestamps (Whisper)
5. OCR on frames → text boxes with confidence (EasyOCR)
6. **Merge OCR into ASR segments** based on timestamp alignment
7. Output: JSON with segments containing both ASR and OCR text

**Critical detail**: OCR text is appended to ASR text as `[SCREEN_TEXT] {ocr_content}` in the final segment text, enabling hybrid retrieval.

### RAG Component Architecture

**Abstract Base Classes** (use these for extensibility):
- `BaseEmbedding` → `HuggingFaceEmbedding`
- `BaseVectorStore` → `InMemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`
- `BaseLLM` → `HuggingFaceLLM`

**Chunking Strategies** (`rag_library/chuncking/`):
- `SimpleTextSplitter` - Character-based with overlap
- `SentenceTextSplitter` - Sentence-boundary aware
- `SemanticChunker` - Embedding similarity-based
- **`ASRTimestampChunker`** - Video-specific: groups segments into time windows (e.g., 15s) with overlap (e.g., 3s)

**RAGPipeline** (`rag_library/pipeline/rag_pipeline.py`):
```python
# Main orchestrator - uses composition pattern
rag = RAGPipeline(
    embedder=HuggingFaceEmbedding("BAAI/bge-base-en-v1.5"),
    vector_store=FaissVectorStore(dim=768),  # Must match embedder dimension
    llm=HuggingFaceLLM("Qwen/Qwen3-0.6B", is_chat_model=True),
    chunker=ASRTimestampChunker(max_seconds=15.0, overlap_seconds=3.0),
    default_top_k=5
)

# Add video segments (not regular documents)
rag.add_segments(segments, document_id="video_id")

# Query
result = rag.answer_question("What is LSTM?", top_k=50, return_sources=True)
```

## Critical Implementation Details

### 1. Timestamp-Aware Chunking
`ASRTimestampChunker` preserves temporal information in chunk metadata:
```python
chunk.metadata = {
    'start_time': 12.5,
    'end_time': 27.5,
    'segment_ids': [5, 6, 7, 8]  # Original segment indices
}
```

This allows answers to reference specific time ranges in the video.

### 2. Hybrid Content Retrieval
Segments combine spoken words + on-screen text:
```python
# In utils.py: load_asr_ocr_segments_from_json()
if seg.get("ocr_text"):
    combined_text = f"{asr_text} [SCREEN_TEXT] {ocr_text}"
```

When querying about visual content (diagrams, slides), OCR text enables retrieval even if not spoken.

### 3. Embedding Dimension Consistency
**Always match vector store dimension to embedder**:
```python
embedder = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")  # 768 dims
store = FaissVectorStore(dim=embedder.dimension)  # Must be 768
```

### 4. LLM System Prompt Pattern
Ground responses in video content:
```python
system_prompt = "You are a helpful assistant that uses ONLY the provided video transcript context."
```

This prevents hallucination beyond retrieved chunks.

### 5. Video Source Handling
`VideoProcessor.process_video()` auto-detects input type:
- Local file: Processes directly from `videos/` directory
- URL: Downloads via `VideoDownloader` first, then processes

### 6. OCR Preprocessing Pipeline
`OCRExtractor` applies image enhancements before text detection:
1. CLAHE (contrast-limited adaptive histogram equalization)
2. Sharpening kernel
3. Adaptive thresholding
4. Confidence filtering (configurable threshold)

This significantly improves OCR accuracy on low-quality video frames.

## Directory Structure

```
VideoRag/
├── main.py                    # Entry point - CLI interactive mode
├── streamlit_app.py          # Streamlit web interface
├── Launcher.py               # Conda environment launcher (legacy)
├── setup.bat                 # Initial environment setup
├── run.bat                   # Run CLI version
├── run_streamlit.bat         # Run Streamlit web app
├── scripts/                  # Video processing layer
│   ├── video_processor.py    # Main orchestrator
│   ├── video_downloader.py   # yt-dlp + HTTP download
│   ├── audio_extractor.py    # FFmpeg audio → WAV
│   ├── frame_extractor.py    # FFmpeg frame extraction
│   ├── transcriber.py        # Whisper wrapper
│   └── ocr_extractor.py      # EasyOCR with preprocessing
├── rag_library/              # RAG framework (reusable)
│   ├── test_chunkers.py      # Interactive chunker testing tool
│   ├── core/
│   │   ├── Models.py         # Data structures
│   │   └── utils.py          # load_asr_ocr_segments_from_json()
│   ├── embeddings/           # BaseEmbedding + implementations
│   ├── llm/                  # BaseLLM + implementations
│   ├── vectordb/             # BaseVectorStore + FAISS/Chroma/InMemory
│   ├── chuncking/            # Chunking strategies (ASRTimestampChunker!)
│   ├── retriever/            # SimpleRetriever
│   ├── pipeline/             # RAGPipeline (main interface)
│   └── loaders/              # PDF/PPTX loaders (non-video docs)
├── videos/                   # Input video directory
├── extracted/                # Intermediate outputs
│   ├── audio/               # WAV files
│   ├── frames/              # Extracted frame images
│   └── ocr/                 # OCR debug outputs
└── transcripts/             # Final JSON outputs
    └── {video_name}_complete.json
```

## Common Pitfalls

### 1. FFmpeg Dependency
Audio/frame extraction requires FFmpeg in system PATH. If missing:
```bash
# Windows
choco install ffmpeg
# Or download from ffmpeg.org
```

### 2. CUDA/CPU Mismatch
`requirements.txt` uses `faiss-cpu`. For GPU:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

Similarly, ensure PyTorch CUDA version matches system CUDA.

### 3. Whisper Model Size vs Performance
- `tiny`: Fast, less accurate
- `base`: Balanced (default)
- `large`: Slow, best accuracy

Choose based on accuracy requirements and hardware.

### 4. Video Download Failures
If yt-dlp fails:
```bash
pip install --upgrade yt-dlp
```

Some platforms require authentication or have rate limits.

### 5. OCR Language Support
EasyOCR downloads models on first use. For new languages:
```python
OCR_LANGUAGES = ['en', 'ar', 'zh']  # Downloads models if not cached
```

Ensure sufficient disk space (~100MB per language).

## Testing and Development

### Testing Chunking Strategies
Use `rag_library/test_chunkers.py` to experiment with different chunking strategies on various document types:

```bash
python rag_library/test_chunkers.py
```

This interactive tool allows you to:
- Choose input type (PDF, PPTX, or Video JSON)
- Select chunking strategy (SimpleTextSplitter, SentenceTextSplitter, SemanticChunker, ASRTimestampChunker)
- Test RAG Q&A on the indexed content
- Compare how different chunkers affect retrieval quality

### Debugging Video Processing
Intermediate outputs are saved for inspection:
- `extracted/audio/{video_name}.wav` - Extracted audio
- `extracted/frames/{video_name}/frame_*.jpg` - Extracted frames
- `extracted/ocr/{video_name}/` - OCR debug outputs
- `transcripts/{video_name}_complete.json` - Final ASR+OCR segments

Check these directories if processing fails or produces unexpected results.

## Extending the System

### Adding a New Embedding Model
```python
# In rag_library/embeddings/
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model_name="text-embedding-3-small"):
        import openai
        self.client = openai.Client()
        self._model = model_name
        self._dimension = 1536  # Model-specific

    def encode(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self._model
        )
        return np.array([e.embedding for e in response.data])
```

### Adding a New Vector Store
```python
# In rag_library/vectordb/
class PineconeVectorStore(BaseVectorStore):
    def __init__(self, index_name: str, dimension: int):
        import pinecone
        # Implementation...

    def add_chunks(self, chunks: List[Chunk]) -> None:
        # Store in Pinecone

    def similarity_search_by_vector(self, vector: np.ndarray, top_k: int):
        # Query Pinecone
```

### Adding a New Chunker
```python
# In rag_library/chuncking/
class FixedTokenChunker:
    def __init__(self, max_tokens=512, overlap_tokens=50):
        import tiktoken
        self.encoder = tiktoken.get_encoding("cl100k_base")
        # Implementation...

    def split_document(self, doc: Document) -> List[Chunk]:
        # Token-based chunking logic
```

## Data Flow Summary

```
User provides video URL/file
    ↓
VideoProcessor.process_video()
    ↓
JSON with ASRSegments (transcripts/{name}_complete.json)
    ↓
load_asr_ocr_segments_from_json() → List[ASRSegment]
    ↓
RAGPipeline.add_segments()
    ↓ (internally)
    1. ASRTimestampChunker.chunk_segments() → Chunks
    2. HuggingFaceEmbedding.encode() → Add embeddings
    3. FaissVectorStore.add_chunks() → Index
    ↓
RAGPipeline.answer_question()
    ↓ (internally)
    1. Embed query
    2. SimpleRetriever.retrieve() → Top-k chunks
    3. Build prompt with context
    4. HuggingFaceLLM.generate() → Answer
```

## Dependencies

**Required system packages**:
- FFmpeg (audio/video processing)
- Python 3.8+

**Key Python packages** (see requirements.txt):
- `torch` - Deep learning framework
- `transformers` - HuggingFace models
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector indexing
- `openai-whisper` - Audio transcription
- `easyocr` - OCR engine
- `yt-dlp` - Video downloading

## File Formats

### Video Processing Output (JSON)
```json
{
  "video_name": "lecture_video",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Welcome to the lecture on neural networks",
      "ocr_text": "LECTURE 1: Neural Networks"
    },
    ...
  ]
}
```

### RAG Answer Output
```python
{
  "answer": "The forget gate in LSTM controls which information...",
  "sources": [  # If return_sources=True
    {
      "chunk_id": "lstm_chunk_12",
      "document_id": "lstm",
      "score": 0.8234,
      "metadata": {"start_time": 45.2, "end_time": 60.2}
    },
    ...
  ]
}
```
