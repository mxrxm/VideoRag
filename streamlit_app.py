import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil
import json
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import RAG library components
from rag_library.embeddings import HuggingFaceEmbedding
from rag_library.vectordb import InMemoryVectorStore, FaissVectorStore, ChromaVectorStore
from rag_library.llm import HuggingFaceLLM
from rag_library.pipeline import RAGPipeline
from rag_library.chuncking import (
    SimpleTextSplitter,
    SentenceTextSplitter,
    SemanticChunker,
    ASRTimestampChunker
)
from rag_library.loaders import load_pdf_as_documents, load_pptx_as_documents, load_asr_ocr_segments_from_json
from scripts.video_processor import VideoProcessor

# Page config
st.set_page_config(
    page_title="VideoRAG - Intelligent Document & Video Q&A",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'indexed_content' not in st.session_state:
    st.session_state.indexed_content = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Model configurations
EMBEDDING_MODELS = {
    "BGE Base (Recommended)": "BAAI/bge-base-en-v1.5",
    "BGE Small": "BAAI/bge-small-en-v1.5",
    "BGE Large": "BAAI/bge-large-en-v1.5",
    "MiniLM L6": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniLM L12": "sentence-transformers/all-MiniLM-L12-v2",
    "MPNet Base": "sentence-transformers/all-mpnet-base-v2",
}

LLM_MODELS = {
    "Qwen 0.6B (Fast)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen 1.8B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Phi-2 (2.7B)": "microsoft/phi-2",
    "TinyLlama (1.1B)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama 8b": "meta-llama/Llama-3.1-8B",
}

VECTOR_STORES = {
    "FAISS (Recommended)": "faiss",
    "ChromaDB": "chroma",
    "In-Memory": "inmemory"
}

CHUNKERS = {
    "Simple Text Splitter": "simple",
    "Sentence Splitter": "sentence",
    "Semantic Chunker": "semantic",
    "ASR Timestamp Chunker (Video Only)": "asr_timestamp"
}


def create_embedder(model_name):
    """Create embedding model"""
    with st.spinner(f"Loading embedding model: {model_name}..."):
        return HuggingFaceEmbedding(model_name)


def create_vector_store(store_type, dimension):
    """Create vector store based on selection"""
    if store_type == "faiss":
        return FaissVectorStore(dim=dimension)
    elif store_type == "chroma":
        return ChromaVectorStore(collection_name="videorag_collection")
    else:  # inmemory
        return InMemoryVectorStore(dim=dimension)


def create_chunker(chunker_type, embedder=None):
    """Create chunker based on selection"""
    if chunker_type == "sentence":
        return SentenceTextSplitter(max_sentences=4, overlap_sentences=1)
    elif chunker_type == "semantic":
        if embedder is None:
            st.error("Semantic chunker requires embedder to be initialized first")
            return SimpleTextSplitter(max_chars=800, overlap=120)
        return SemanticChunker(embedder=embedder, similarity_threshold=0.7)
    elif chunker_type == "asr_timestamp":
        return ASRTimestampChunker(max_seconds=15.0, overlap_seconds=3.0)
    else:  # simple
        return SimpleTextSplitter(max_chars=800, overlap=120)


def create_llm(model_name):
    """Create LLM model"""
    with st.spinner(f"Loading LLM: {model_name}..."):
        return HuggingFaceLLM(
            model_name=model_name,
            device="auto",
            torch_dtype="auto",
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            is_chat_model=True,
            system_prompt="You are a helpful assistant that uses ONLY the provided context to answer questions accurately."
            
        )


def build_rag_pipeline(embed_model, llm_model, vector_store_type, chunker_type,
                       summary_llm_model=None, gen_llm_model=None):
    """Build complete RAG pipeline with optional specialized models"""
    try:
        # Create embedder
        embedder = create_embedder(embed_model)

        # Create vector store
        vector_store = create_vector_store(vector_store_type, embedder.dimension)

        # Create chunker (for semantic chunker, pass embedder)
        chunker = create_chunker(chunker_type, embedder if chunker_type == "semantic" else None)

        # Create main LLM
        llm = create_llm(llm_model)
        
        # Create optional specialized LLMs
        summary_llm = None
        if summary_llm_model:
            summary_llm = create_llm(summary_llm_model)
        
        generation_llm = None
        if gen_llm_model:
            generation_llm = create_llm(gen_llm_model)

        # Build pipeline
        rag = RAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            chunker=chunker,
            default_top_k=5,
            summary_llm=summary_llm,
            generation_llm=generation_llm
        )

        return rag, True
    except Exception as e:
        st.error(f"Error building RAG pipeline: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, False


def process_pdf(uploaded_file, rag_pipeline):
    """Process uploaded PDF file"""
    with st.spinner("Processing PDF..."):
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            # Load and index
            docs = load_pdf_as_documents(tmp_path, per_page=True)
            if docs:
                rag_pipeline.add_documents(docs)
                st.success(f"✅ Indexed {len(docs)} pages from PDF")
                return len(docs)
            else:
                st.warning("No text found in PDF")
                return 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def process_pptx(uploaded_file, rag_pipeline):
    """Process uploaded PPTX file"""
    with st.spinner("Processing PowerPoint..."):
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            # Load and index
            docs = load_pptx_as_documents(tmp_path, per_slide=True)
            if docs:
                rag_pipeline.add_documents(docs)
                st.success(f"✅ Indexed {len(docs)} slides from PowerPoint")
                return len(docs)
            else:
                st.warning("No text found in PowerPoint")
                return 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def process_video(uploaded_file, rag_pipeline, use_ocr=True, ocr_languages=['en'], language=None):
    """Process uploaded video file"""
    with st.spinner("Processing video... This may take several minutes..."):
        # Save to videos directory
        video_path = Path("videos") / uploaded_file.name
        video_path.parent.mkdir(exist_ok=True)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Initialize video processor
            processor = VideoProcessor(
                whisper_model="base",
                project_root=".",
                delete_original=False,
                use_ocr=use_ocr,
                ocr_languages=ocr_languages
            )

            # Process video
            status_placeholder = st.empty()
            status_placeholder.info("📹 Extracting audio and frames...")

            results_path, result = processor.process_video(
                video_filename=uploaded_file.name,
                fps=1,
                language=language,
                ocr_confidence=0.5
            )

            status_placeholder.info("🔤 Loading and indexing segments...")

            # Load segments and index
            segments = load_asr_ocr_segments_from_json(results_path)
            if segments:
                rag_pipeline.add_segments(segments, document_id=uploaded_file.name)
                status_placeholder.success(f"✅ Indexed {len(segments)} segments from video")
                return len(segments)
            else:
                status_placeholder.warning("No segments found in video")
                return 0
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return 0


def process_video_url(video_url, rag_pipeline, use_ocr=True, ocr_languages=['en'], language=None):
    """Process video from URL"""
    with st.spinner("Downloading and processing video... This may take several minutes..."):
        try:
            # Initialize video processor
            processor = VideoProcessor(
                whisper_model="base",
                project_root=".",
                delete_original=False,
                use_ocr=use_ocr,
                ocr_languages=ocr_languages
            )

            # Process video
            status_placeholder = st.empty()
            status_placeholder.info("⬇️ Downloading video from URL...")

            results_path, result = processor.process_video(
                video_filename=video_url,
                fps=1,
                language=language,
                ocr_confidence=0.5
            )

            status_placeholder.info("🔤 Loading and indexing segments...")

            # Load segments and index
            video_name = result.get('video_name', 'video')
            segments = load_asr_ocr_segments_from_json(results_path)
            if segments:
                rag_pipeline.add_segments(segments, document_id=video_name)
                status_placeholder.success(f"✅ Indexed {len(segments)} segments from video")
                return len(segments), video_name
            else:
                status_placeholder.warning("No segments found in video")
                return 0, video_name
        except Exception as e:
            st.error(f"Error processing video URL: {str(e)}")
            return 0, None


# App Header
st.title("🎥 VideoRAG - Intelligent Document & Video Q&A")
st.markdown("Upload documents (PDF, PowerPoint, or Video) and ask questions using advanced RAG technology")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Embedding Model")
    selected_embed = st.selectbox(
        "Choose embedding model",
        options=list(EMBEDDING_MODELS.keys()),
        help="Embedding model converts text to vectors for similarity search"
    )
    embed_model = EMBEDDING_MODELS[selected_embed]

    st.subheader("Language Model (LLM)")
    selected_llm = st.selectbox(
        "Choose LLM for Q&A",
        options=list(LLM_MODELS.keys()),
        help="Main LLM for answering questions"
    )
    llm_model = LLM_MODELS[selected_llm]

    # NEW: Advanced Model Settings
    with st.expander("🎯 Advanced: Specialized Models", expanded=False):
        st.markdown("""
        Use different models for specific tasks:
        - **Summarization**: Use a faster/cheaper model for summaries
        - **Question Generation**: Use a creative model for generating questions
        """)
        
        # Summarization model
        use_summary_model = st.checkbox(
            "Use separate model for summarization",
            value=False,
            help="Enable to use a different model for summaries"
        )
        
        if use_summary_model:
            selected_summary_llm = st.selectbox(
                "Summary Model",
                options=list(LLM_MODELS.keys()),
                key="summary_llm",
                help="Model for generating summaries"
            )
            summary_llm_model = LLM_MODELS[selected_summary_llm]
        else:
            summary_llm_model = None
        
        # Question generation model
        use_gen_model = st.checkbox(
            "Use separate model for question generation",
            value=False,
            help="Enable to use a different model for generating questions"
        )
        
        if use_gen_model:
            selected_gen_llm = st.selectbox(
                "Generation Model",
                options=list(LLM_MODELS.keys()),
                key="gen_llm",
                help="Model for generating questions"
            )
            gen_llm_model = LLM_MODELS[selected_gen_llm]
        else:
            gen_llm_model = None

    st.subheader("Vector Database")
    selected_vector_store = st.selectbox(
        "Choose vector database",
        options=list(VECTOR_STORES.keys()),
        help="Vector database stores and retrieves document embeddings"
    )
    vector_store_type = VECTOR_STORES[selected_vector_store]

    st.subheader("Chunking Strategy")
    selected_chunker = st.selectbox(
        "Choose chunker",
        options=list(CHUNKERS.keys()),
        help="Chunker splits documents into manageable pieces"
    )
    chunker_type = CHUNKERS[selected_chunker]

    st.divider()

    # Video Processing Options
    st.subheader("📹 Video Processing Options")

    use_ocr = st.checkbox(
        "Enable OCR",
        value=True,
        help="Extract text from video frames using OCR"
    )

    # Language selection
    language_options = {
        "Auto-detect": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh",
        "Arabic": "ar",
        "Hindi": "hi"
    }

    selected_language = st.selectbox(
        "Transcription Language",
        options=list(language_options.keys()),
        help="Language for audio transcription (auto-detect recommended)"
    )
    language = language_options[selected_language]

    # OCR languages (only if OCR is enabled)
    if use_ocr:
        ocr_lang_options = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Arabic": "ar",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko"
        }

        selected_ocr_langs = st.multiselect(
            "OCR Languages",
            options=list(ocr_lang_options.keys()),
            default=["English"],
            help="Select languages for OCR text detection"
        )
        ocr_languages = [ocr_lang_options[lang] for lang in selected_ocr_langs]
    else:
        ocr_languages = ['en']

    st.divider()

    # Build/Rebuild pipeline button
    if st.button("🔧 Initialize/Rebuild RAG Pipeline", use_container_width=True):
        st.session_state.rag_pipeline, success = build_rag_pipeline(
            embed_model,
            llm_model,
            vector_store_type,
            chunker_type,
            summary_llm_model,
            gen_llm_model
        )
        if success:
            st.success("✅ RAG Pipeline initialized!")
            
            # Show which models are active
            if st.session_state.rag_pipeline:
                models_info = st.session_state.rag_pipeline.get_active_models()
                with st.expander("ℹ️ Active Models"):
                    st.write(f"**Q&A Model:** {models_info['qa_model']}")
                    st.write(f"**Summary Model:** {models_info['summary_model']}")
                    st.write(f"**Generation Model:** {models_info['generation_model']}")
            
            st.session_state.processing_complete = False
            st.session_state.indexed_content = None
            st.session_state.chat_history = []
        else:
            st.error("❌ Failed to initialize pipeline")

    # Clear all button
    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        st.session_state.rag_pipeline = None
        st.session_state.chat_history = []
        st.session_state.indexed_content = None
        st.session_state.processing_complete = False
        st.success("Cleared!")
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Content")

    # Check if pipeline is initialized
    if st.session_state.rag_pipeline is None:
        st.warning("⚠️ Please initialize the RAG pipeline first using the sidebar configuration!")
    else:
        st.info("✅ RAG Pipeline is ready")

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "🔗 Video URL"])

    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'pptx', 'ppt', 'mp4', 'avi', 'mov', 'mkv'],
            help="Upload PDF, PowerPoint, or Video file"
        )

        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")

            # Process button
            if st.button("🚀 Process & Index File", use_container_width=True, key="process_file"):
                if st.session_state.rag_pipeline is None:
                    st.error("Please initialize the RAG pipeline first!")
                else:
                    # Determine file type and process
                    file_ext = Path(uploaded_file.name).suffix.lower()

                    if file_ext == '.pdf':
                        chunks = process_pdf(uploaded_file, st.session_state.rag_pipeline)
                        if chunks > 0:
                            st.session_state.processing_complete = True
                            st.session_state.indexed_content = uploaded_file.name
                            st.balloons()
                    elif file_ext in ['.pptx', '.ppt']:
                        chunks = process_pptx(uploaded_file, st.session_state.rag_pipeline)
                        if chunks > 0:
                            st.session_state.processing_complete = True
                            st.session_state.indexed_content = uploaded_file.name
                            st.balloons()
                    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        # Check if using appropriate chunker for video
                        if chunker_type != "asr_timestamp":
                            st.warning("⚠️ Consider using 'ASR Timestamp Chunker' for video files for best results")
                        chunks = process_video(
                            uploaded_file,
                            st.session_state.rag_pipeline,
                            use_ocr=use_ocr,
                            ocr_languages=ocr_languages,
                            language=language
                        )
                        if chunks > 0:
                            st.session_state.processing_complete = True
                            st.session_state.indexed_content = uploaded_file.name
                            st.balloons()
                    else:
                        st.error("Unsupported file type")

    with tab2:
        st.markdown("**Enter video URL (YouTube, Vimeo, or direct link)**")

        video_url = st.text_input(
            "Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube URL, Vimeo URL, or direct video link"
        )

        if video_url:
            st.info(f"📹 URL: {video_url}")

            # Process button for URL
            if st.button("🚀 Download & Process Video", use_container_width=True, key="process_url"):
                if st.session_state.rag_pipeline is None:
                    st.error("Please initialize the RAG pipeline first!")
                else:
                    # Check if using appropriate chunker for video
                    if chunker_type != "asr_timestamp":
                        st.warning("⚠️ Consider using 'ASR Timestamp Chunker' for video files for best results")

                    chunks, video_name = process_video_url(
                        video_url,
                        st.session_state.rag_pipeline,
                        use_ocr=use_ocr,
                        ocr_languages=ocr_languages,
                        language=language
                    )

                    if chunks > 0:
                        st.session_state.processing_complete = True
                        st.session_state.indexed_content = video_name if video_name else video_url
                        st.balloons()

# RIGHT COLUMN - INTERACTION
with col2:
    st.header("💬 Interact with Content")

    if not st.session_state.processing_complete:
        st.info("📋 Process a document first to start interacting")
    else:
        st.success(f"✅ Ready to interact with: **{st.session_state.indexed_content}**")

        # Display vector store stats
        if st.session_state.rag_pipeline:
            try:
                stats = st.session_state.rag_pipeline.get_stats()
                st.caption(f"📊 {stats['total_chunks']} chunks | {stats['unique_documents']} documents | {stats['video_chunks']} video segments")
            except:
                vector_store = st.session_state.rag_pipeline.vector_store
                st.caption(f"Vector store contains {len(vector_store)} chunks")

        st.divider()

        # CREATE TABS FOR DIFFERENT FEATURES
        tab_qa, tab_summary, tab_gen_q = st.tabs([
            "💬 Q&A",
            "📝 Summarize",
            "🎲 Generate Questions"
        ])

        # TAB 1: Q&A
        with tab_qa:
            st.subheader("Ask Questions")
            
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**Q{i+1}:** {question}")
                    st.markdown(f"**A{i+1}:** {answer}")
                    st.divider()

            # Question input
            question = st.text_input(
                "Enter your question:",
                key="question_input",
                placeholder="What is this document about?"
            )

            col_ask, col_clear = st.columns([3, 1])

            with col_ask:
                ask_button = st.button("🔍 Ask", use_container_width=True)

            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            if ask_button and question:
                if st.session_state.rag_pipeline:
                    with st.spinner("Thinking..."):
                        try:
                            result = st.session_state.rag_pipeline.answer_question(
                                question,
                                top_k=5,
                                return_sources=True
                            )

                            answer = result["answer"]
                            st.session_state.chat_history.append((question, answer))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())

        # TAB 2: SUMMARIZATION
        with tab_summary:
            st.subheader("📝 Summarize Content")
            
            # Show which model is being used
            if st.session_state.rag_pipeline:
                models_info = st.session_state.rag_pipeline.get_active_models()
                st.caption(f"🤖 Using: {models_info['summary_model']}")
            
            st.markdown("Generate different types of summaries from your indexed content.")
            
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                summary_type = st.selectbox(
                    "Summary Type",
                    options=["Comprehensive", "Brief", "Key Points", "Main Topics"],
                    help="Choose the type of summary"
                )
            
            with col_sum2:
                max_chunks = st.slider(
                    "Max Chunks",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="More chunks = more comprehensive"
                )
            
            st.markdown("---")
            
            if st.button("✨ Generate Summary", use_container_width=True, key="gen_summary"):
                if st.session_state.rag_pipeline:
                    with st.spinner("Generating summary..."):
                        try:
                            type_map = {
                                "Comprehensive": "comprehensive",
                                "Brief": "brief",
                                "Key Points": "keypoints",
                                "Main Topics": "topics"
                            }
                            
                            summary = st.session_state.rag_pipeline.summarize_content(
                                max_chunks=max_chunks,
                                summary_type=type_map[summary_type],
                                max_tokens=1024
                            )
                            
                            st.success("✅ Summary Generated!")
                            st.markdown("### Summary:")
                            st.info(summary)
                            
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary,
                                file_name=f"summary_{st.session_state.indexed_content}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())

        # TAB 3: GENERATE QUESTIONS (Fixed auto-answer)
        with tab_gen_q:
            st.subheader("🎲 Generate Questions")
            
            # Show which model is being used
            if st.session_state.rag_pipeline:
                models_info = st.session_state.rag_pipeline.get_active_models()
                st.caption(f"🤖 Using: {models_info['generation_model']}")
            
            st.markdown("Let AI generate questions based on your content!")
            
            col_gq1, col_gq2 = st.columns(2)
            
            with col_gq1:
                num_questions = st.slider(
                    "Number of questions",
                    min_value=3,
                    max_value=10,
                    value=5,
                    key="num_gen_questions"
                )
            
            with col_gq2:
                question_type = st.selectbox(
                    "Question Type",
                    options=["Comprehension", "Factual", "Analytical"],
                    key="gen_question_type"
                )
            
            with st.expander("ℹ️ Question Type Descriptions"):
                st.markdown("""
                - **Comprehension**: Test understanding of main ideas
                - **Factual**: Test recall of specific facts
                - **Analytical**: Require critical thinking
                """)
            
            st.markdown("---")
            
            # Initialize session state for generated questions
            if 'generated_questions' not in st.session_state:
                st.session_state.generated_questions = []
            
            if st.button("🎲 Generate Questions", use_container_width=True, key="gen_questions"):
                with st.spinner("Generating questions..."):
                    try:
                        type_map = {
                            "Comprehension": "comprehension",
                            "Factual": "factual",
                            "Analytical": "analytical"
                        }
                        
                        questions = st.session_state.rag_pipeline.generate_questions(
                            num_questions=num_questions,
                            question_type=type_map[question_type]
                        )
                        
                        # Store in session state
                        st.session_state.generated_questions = questions
                        
                        if questions:
                            st.success(f"✅ Generated {len(questions)} questions!")
                        else:
                            st.warning("⚠️ Could not generate questions.")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            
            # Display generated questions if they exist
            if st.session_state.generated_questions:
                st.markdown("### Generated Questions:")
                for i, q in enumerate(st.session_state.generated_questions, 1):
                    st.markdown(f"**{i}.** {q}")
                
                st.markdown("---")
                
                # Two columns for buttons
                col_answer, col_download = st.columns(2)
                
                with col_answer:
                    if st.button("📝 Answer All Questions", use_container_width=True, key="answer_all_btn"):
                        with st.spinner("Generating answers for all questions..."):
                            try:
                                results = st.session_state.rag_pipeline.answer_multiple_questions(
                                    st.session_state.generated_questions,
                                    top_k=3
                                )
                                
                                st.success("✅ All questions answered!")
                                
                                # Display results
                                st.markdown("### Answers:")
                                for i, result in enumerate(results, 1):
                                    with st.expander(f"Q{i}: {result['question']}", expanded=True):
                                        st.markdown("**Answer:**")
                                        st.info(result['answer'])
                                
                                # Prepare download text
                                qa_text = "\n\n".join([
                                    f"Q{i}: {r['question']}\n\nA: {r['answer']}\n{'-'*60}"
                                    for i, r in enumerate(results, 1)
                                ])
                                
                                # Download button for Q&A
                                st.download_button(
                                    label="📥 Download Questions & Answers",
                                    data=qa_text,
                                    file_name=f"generated_qa_{st.session_state.indexed_content}.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                    key="download_qa_btn"
                                )
                                
                            except Exception as e:
                                st.error(f"Error answering questions: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
                
                with col_download:
                    # Download just questions
                    questions_text = "\n".join([
                        f"{i}. {q}" 
                        for i, q in enumerate(st.session_state.generated_questions, 1)
                    ])
                    st.download_button(
                        label="📥 Download Questions Only",
                        data=questions_text,
                        file_name=f"generated_questions_{st.session_state.indexed_content}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_q_only_btn"
                    )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>VideoRAG - Powered by HuggingFace Transformers, FAISS/ChromaDB, and Whisper</small>
</div>
""", unsafe_allow_html=True)