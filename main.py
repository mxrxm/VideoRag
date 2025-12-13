import os
import sys
from pathlib import Path
import rag_library
# from rag_library.loaders import load_pdf_as_documents
from rag_library.embeddings import HuggingFaceEmbedding
from rag_library.vectordb import InMemoryVectorStore ,FaissVectorStore,ChromaVectorStore
from rag_library.llm import HuggingFaceLLM
from rag_library.pipeline import RAGPipeline
from rag_library.chuncking import SemanticChunker,SimpleTextSplitter,ASRTimestampChunker,SentenceTextSplitter
from rag_library.loaders import load_pdf_as_documents, load_pptx_as_documents,load_asr_ocr_segments_from_json
# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from scripts.video_processor import VideoProcessor

# Configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
FPS = 1                   # Frames per second to extract
LANGUAGE = None           # Language code ('en', 'es', etc.) or None for auto-detect
DELETE_ORIGINAL = False   # Delete videos after processing?
USE_OCR = True            # Enable OCR on frames?
OCR_LANGUAGES = ['en']    # OCR languages: ['en', 'ar', 'fr', 'zh', etc.]
OCR_CONFIDENCE = 0.5      # Minimum OCR confidence (0-1)

# Initialize the processor
processor = VideoProcessor(
    whisper_model=WHISPER_MODEL,
    project_root=".",
    delete_original=DELETE_ORIGINAL,
    use_ocr=USE_OCR,
    ocr_languages=OCR_LANGUAGES
)

def init():
    """Main pipeline controller"""
    
    print("="*60)
    print("VIDEO RAG PIPELINE")
    print("="*60)
    videos_dir = Path("videos")
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv"]
    video_files = []  

    for ext in video_extensions:
        video_files.extend(videos_dir.glob(ext))
    # Check for videos

    

    
    # Display available videos (if any)
    # if video_files:
    #     print(f"\n{'='*60}")
    #     print("AVAILABLE LOCAL VIDEOS")
    #     print(f"{'='*60}")
    #     for i, video in enumerate(video_files, 1):
    #         print(f"{i}. {video.name}")
    #     print(f"{'='*60}\n")
    # else:
    #     print("\nℹ️  No local videos found in 'videos/' directory")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Process from URL (YouTube, Vimeo, or direct video link)")
    print("2. Process a specific local video")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Process from URL
        process_from_url()
    
    elif choice == "2":
        # Process specific local video
        if video_files:
            print(f"\n{'='*60}")
            print("AVAILABLE LOCAL VIDEOS")
            print(f"{'='*60}")
            for i, video in enumerate(video_files, 1):
                print(f"{i}. {video.name}")
            print(f"{'='*60}\n")
        if not video_files:
            print("\n No local videos found in 'videos/' directory")
            return
        
        video_num = int(input(f"Enter video number (1-{len(video_files)}): "))
        if 1 <= video_num <= len(video_files):
            video_to_process = video_files[video_num - 1].name
            
            try:
                results_path,result = processor.process_video(
                    video_filename=video_to_process,
                    fps=FPS,
                    language=LANGUAGE,
                    ocr_confidence=OCR_CONFIDENCE
                )
                return results_path
        
            except Exception as e:
                print(f"\nError: {e}")
        else:
            print("Invalid video number!")

    
    elif choice == "3":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")


def process_from_url():
    """Process video from URL"""
    print("\n" + "="*60)
    print("PROCESS VIDEO FROM URL")
    print("="*60)
    
    print("\nSupported sources:")
    print("  • YouTube: https://www.youtube.com/watch?v=VIDEO_ID")
    print("  • Vimeo: https://vimeo.com/VIDEO_ID")
    print("  • Direct links: https://example.com/video.mp4")
    print("  • And many more platforms!")
    
    video_url = input("\nEnter video URL: ").strip()
    
    if not video_url:
        print(" No URL provided!")
        return
    
    # Ask for custom filename (optional)
    custom_name = input("\nCustom filename (optional, press Enter to skip): ").strip()
    custom_filename = custom_name if custom_name else None
    
    # Ask for processing options
    print("\n Processing Options:")
    print(f"   FPS: {FPS} (frames per second)")
    print(f"   Language: {LANGUAGE if LANGUAGE else 'Auto-detect'}")
    print(f"   OCR: {'Enabled' if USE_OCR else 'Disabled'}")
    print(f"   Delete after: {'Yes' if DELETE_ORIGINAL else 'No'}")
    
    proceed = input("\nProceed with these settings? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("Cancelled.")
        return
    
    print(f"\n{'='*60}")
    print("DOWNLOADING AND PROCESSING...")
    print(f"{'='*60}\n")
    
    try:
        results_path,result = processor.process_video(
            video_filename=video_url,  # processor handles URLs automatically
            fps=FPS,
            language=LANGUAGE,
            ocr_confidence=OCR_CONFIDENCE,
            custom_filename=custom_filename
        )
        
        print("\n SUCCESS!")
        
        # Additional info for downloaded videos
        if result.get('was_downloaded'):
            print(f"\n Video was downloaded to: videos/{result['video_name']}.*")
            if DELETE_ORIGINAL:
                print(" Video was deleted after processing (as configured)")
        return results_path
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Install yt-dlp: pip install yt-dlp --break-system-packages")
        print("  2. Check if the URL is accessible")
        print("  3. Check your internet connection")
        print("  4. Try a direct video URL instead")


# def process_single_video_direct(video_source):
#     """
#     Direct processing without prompts (for automation)
    
#     Args:
#         video_source: Can be a local filename OR a URL
#     """
#     try:
#         result = processor.process_video(
#             video_filename=video_source,  # Works with both local files and URLs
#             fps=FPS,
#             language=LANGUAGE,
#             ocr_confidence=OCR_CONFIDENCE
#         )
#         return result
#     except Exception as e:
#         print(f"Error: {e}")
#  
# 
    #    return None


#--------------------------------------------------------------------------------------------------------------------------------
def build_core_components(
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "Qwen/Qwen3-0.6B",
    llm_is_chat: bool = True,
):
    embedder = HuggingFaceEmbedding(embed_model)
    store = FaissVectorStore(dim=embedder.dimension)

    llm = HuggingFaceLLM(
        model_name=llm_model,
        device="auto",          # or "cpu"
        torch_dtype="auto",
        max_new_tokens=512,     # keep small for testing
        temperature=0.7,
        top_p=0.95,
        is_chat_model=llm_is_chat,
        system_prompt="You are a helpful assistant that uses ONLY the provided context.",
    )
    return embedder, store, llm


def choose_input_type() -> str:
    print("\nChoose input type:")
    print("1) PDF")
    print("2) PPTX")
    print("3) Video JSON (ASR+OCR segments)")
    choice = input("Enter 1/2/3: ").strip()

    mapping = {"1": "pdf", "2": "pptx", "3": "video_json"}
    if choice not in mapping:
        print("Invalid choice. Defaulting to PDF.")
        return "pdf"
    return mapping[choice]


def choose_chunker_text(embedder: HuggingFaceEmbedding):
    """
    Chunkers for PDF/PPTX documents.
    Video JSON uses ASRTimestampChunker separately.
    """
    print("\nChoose chunker for text docs:")
    print("1) SimpleTextSplitter")
    print("2) SentenceTextSplitter")
    print("3) SemanticChunker")
    choice = input("Enter 1/2/3: ").strip()

    if choice == "2":
        return SentenceTextSplitter(max_sentences=4, overlap_sentences=1)
    if choice == "3":
        return SemanticChunker(embedder=embedder, similarity_threshold=0.7)
    return SimpleTextSplitter(max_chars=800, overlap=120)


def run_pdf_flow(rag: RAGPipeline, pdf_path: str):
    docs = load_pdf_as_documents(pdf_path, per_page=True)
    print(f"\nLoaded {len(docs)} Document(s) from PDF.")
    if not docs:
        print("No text found in PDF.")
        return
    print("Indexing PDF...")
    rag.add_documents(docs)
    print("Done indexing PDF.")


def run_pptx_flow(rag: RAGPipeline, pptx_path: str):
    docs = load_pptx_as_documents(pptx_path, per_slide=True)
    print(f"\nLoaded {len(docs)} Document(s) from PPTX.")
    if not docs:
        print("No text found in PPTX.")
        return
    print("Indexing PPTX...")
    rag.add_documents(docs)
    print("Done indexing PPTX.")


def run_video_json_flow(rag: RAGPipeline, json_path: str, document_id: str = "video"):
    """
    Video JSON → segments → ASRTimestampChunker → chunks → index
    This assumes you already added `rag.add_segments(...)` to your pipeline.
    """
    segments = load_asr_ocr_segments_from_json(json_path)  # your function
    print(f"\nLoaded {len(segments)} segment(s) from JSON.")
    if not segments:
        print("No segments found. Check JSON structure.")
        return

    print("Indexing Video segments...")
    rag.add_segments(segments, document_id=document_id)
    print("Done indexing video.")


def interactive_qa(rag: RAGPipeline):
    print("\n=== RAG Q&A (type 'exit' to quit) ===")
    while True:
        q = input("\nQuestion: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        result = rag.answer_question(q, top_k=5, return_sources=True)

        print("\nAnswer:\n", result["answer"])

def main(path):
   # 1) choose input type
    input_type = choose_input_type()

    # 2) build core components
    # If you want to default to bge embeddings for everything:
    # embed_model = "BAAI/bge-base-en-v1.5"
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedder, store, llm = build_core_components(
        embed_model=embed_model,
        llm_model="Qwen/Qwen3-0.6B",
        llm_is_chat=True,
    )

    # 3) choose chunker for text docs
    if input_type in {"pdf", "pptx"}:
        chunker = choose_chunker_text(embedder)
    else:
        # video uses timestamp chunker
        chunker = ASRTimestampChunker(max_seconds=15.0, overlap_seconds=3.0)

    # 4) build pipeline
    rag = RAGPipeline(
        embedder=embedder,
        vector_store=store,
        llm=llm,
        chunker=chunker,
        default_top_k=5,
    )

    # 5) run chosen flow
    if input_type == "pdf":
        path = input("\nEnter PDF path: ").strip().strip('"')
        run_pdf_flow(rag, path)

    elif input_type == "pptx":
        path = input("\nEnter PPTX path: ").strip().strip('"')
        run_pptx_flow(rag, path)

    else:  # video_json
        # path = input("\nEnter Video JSON path: ").strip().strip('"')
        doc_id = input("Enter document_id (default=video): ").strip() or "video"
        run_video_json_flow(rag, path, document_id=doc_id)

    print(f"\nVector store size: {len(store)} chunks")
    interactive_qa(rag) 

if __name__ == "__main__":
    # Run interactive mode (default)
    path=init()
    main(path)
    
    # Or run direct mode with a specific video (comment main() above and uncomment below)
    # result = process_single_video_direct("video.mp4")  # Local file
    # result = process_single_video_direct("https://youtube.com/watch?v=ID")  # URL
    
    # Or run quick URL test (uncomment below)
    # quick_url_test()