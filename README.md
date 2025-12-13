# VideoRAG 🎥

**Intelligent Document & Video Q&A System using Retrieval-Augmented Generation**

VideoRAG is a powerful system that extracts content from videos (audio transcription + on-screen text OCR) and documents (PDF/PowerPoint) to enable semantic question-answering through advanced RAG technology.

## ✨ Features

- 🎥 **Multi-Modal Video Processing**: Extract both audio (Whisper ASR) and visual text (EasyOCR)
- 📄 **Document Support**: Process PDF and PowerPoint files
- 🌐 **Web Interface**: User-friendly Streamlit UI with drag-and-drop uploads
- 🔧 **Configurable Components**: Choose your embedding models, LLMs, vector databases, and chunking strategies
- 💬 **Interactive Q&A**: Ask questions about your content with chat history
- 🎯 **Timestamp-Aware**: Answers reference specific time ranges in videos
- 🚀 **Easy Setup**: One-click installation scripts for Windows

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run setup script (installs dependencies and creates virtual environment)
setup.bat
```

### 2. Launch Streamlit Web Interface (Recommended)

```bash
# Start the web application
run_streamlit.bat
```

The app will open automatically in your browser at `http://localhost:8501`

### 3. Use the Application

1. **Configure** (Sidebar):
   - Select Embedding Model (default: BGE Base)
   - Select LLM (default: Qwen 0.6B)
   - Select Vector Database (default: FAISS)
   - Select Chunking Strategy (default: Simple)
   - Click "Initialize RAG Pipeline"

2. **Upload** (Main Area):
   - Drag and drop your file (PDF, PPTX, or Video)
   - Click "Process & Index"
   - Wait for processing to complete

3. **Ask Questions**:
   - Type your question in the chat interface
   - Click "Ask"
   - Get AI-powered answers based on your content!

## 🖥️ Alternative: Command Line Interface

```bash
# Interactive CLI mode
run.bat
# Or: python main.py

# Test different chunking strategies
python rag_library/test_chunkers.py
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **FFmpeg**: Required for video/audio processing ([Download](https://ffmpeg.org/))
- **RAM**: 8GB+ recommended (16GB for large models)
- **Disk Space**: 5GB+ for models and dependencies

### Python Dependencies
All dependencies are installed automatically via `setup.bat`. Key packages include:
- `torch` - Deep learning framework
- `transformers` - HuggingFace models
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector indexing
- `openai-whisper` - Audio transcription
- `easyocr` - OCR engine
- `streamlit` - Web interface
- `chromadb` - Alternative vector store

## 🎯 Supported File Types

| Type | Extensions | Processing |
|------|-----------|------------|
| **Video** | `.mp4`, `.avi`, `.mov`, `.mkv` | Whisper ASR + EasyOCR |
| **PDF** | `.pdf` | PyMuPDF text extraction |
| **PowerPoint** | `.pptx`, `.ppt` | python-pptx extraction |

## 🔧 Configuration Options

### Embedding Models
- **BGE Base** (Recommended) - Balanced performance
- **BGE Small** - Faster, less accurate
- **BGE Large** - More accurate, slower
- **MiniLM L6** - Lightweight option
- **MPNet Base** - General-purpose

### Language Models
- **Qwen 0.6B** (Default) - Fast, good quality
- **Qwen 1.8B** - Better quality, slower
- **Phi-2** - Microsoft model
- **TinyLlama** - Lightweight option

### Vector Databases
- **FAISS** (Recommended) - Fast, efficient
- **ChromaDB** - Persistent storage
- **In-Memory** - Fastest, no persistence

### Chunking Strategies
- **Simple Text Splitter** - Character-based chunking
- **Sentence Splitter** - Sentence boundary-aware
- **Semantic Chunker** - Similarity-based chunking
- **ASR Timestamp Chunker** - Time-windowed (for videos)

## 📂 Project Structure

```
VideoRag/
├── streamlit_app.py          # Web interface (main entry point)
├── main.py                   # CLI interface
├── setup.bat                 # Environment setup script
├── run_streamlit.bat         # Launch web app
├── run.bat                   # Launch CLI
├── scripts/                  # Video processing pipeline
│   ├── video_processor.py    # Main orchestrator
│   ├── video_downloader.py   # YouTube/URL download
│   ├── audio_extractor.py    # FFmpeg audio extraction
│   ├── frame_extractor.py    # FFmpeg frame extraction
│   ├── transcriber.py        # Whisper ASR
│   └── ocr_extractor.py      # EasyOCR
├── rag_library/              # RAG framework
│   ├── embeddings/           # Embedding models
│   ├── llm/                  # Language models
│   ├── vectordb/             # Vector databases
│   ├── chuncking/            # Chunking strategies
│   ├── retriever/            # Retrieval logic
│   ├── pipeline/             # RAG orchestration
│   └── loaders/              # Document loaders
├── videos/                   # Input videos
├── extracted/                # Intermediate outputs
│   ├── audio/
│   ├── frames/
│   └── ocr/
└── transcripts/              # Final JSON outputs
```

## 🎬 How It Works

### For Videos:
1. **Extract Audio** → Convert to WAV via FFmpeg
2. **Extract Frames** → Sample frames at 1 FPS
3. **Transcribe** → Whisper generates timestamped text
4. **OCR** → EasyOCR extracts on-screen text
5. **Merge** → Combine ASR + OCR into segments
6. **Chunk** → Split into time windows (default: 15s)
7. **Embed** → Convert to vectors
8. **Index** → Store in FAISS/ChromaDB
9. **Query** → Retrieve relevant chunks → Generate answer

### For Documents:
1. **Load** → Extract text from PDF/PPTX
2. **Chunk** → Split by sentences/semantics
3. **Embed** → Convert to vectors
4. **Index** → Store in vector database
5. **Query** → Retrieve → Generate answer

## 🐛 Troubleshooting

### FFmpeg Not Found
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/
```

### Out of Memory
- Use smaller models (Qwen 0.6B, MiniLM L6)
- Reduce FPS for video processing
- Use CPU instead of GPU

### Slow Processing
- Use smaller Whisper model (`tiny` instead of `base`)
- Reduce FPS for frame extraction
- Use FAISS instead of ChromaDB

### Video Download Fails
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

## 📖 Documentation

For detailed architecture and development guidelines, see [CLAUDE.md](CLAUDE.md)

## 🙏 Acknowledgments

Built with:
- [Whisper](https://github.com/openai/whisper) - OpenAI's speech recognition
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Text detection
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook's vector search
- [HuggingFace Transformers](https://huggingface.co/transformers) - NLP models
- [Streamlit](https://streamlit.io/) - Web framework

## 📄 License

This project is for educational purposes.

---

**Made with ❤️ for intelligent document and video understanding**
