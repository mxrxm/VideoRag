import os
import sys
from pathlib import Path
import rag_library
# from rag_library.loaders import load_pdf_as_documents
from rag_library.embeddings import HuggingFaceEmbedding
from rag_library.vectordb import InMemoryVectorStore ,FaissVectorStore
from rag_library.llm import HuggingFaceLLM
from rag_library.pipeline import RAGPipeline
from rag_library.chuncking import SemanticChunker,SimpleTextSplitter,ASRTimestampChunker
from rag_library.core.utils import load_asr_ocr_segments_from_json
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
        print("❌ No URL provided!")
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
#         return None



def main():
    results_path=init()
    pdf_path = r"C:\Users\Fatma PC\Desktop\raglib\1706.03762v7.pdf"  # your Transformer paper
    json_path=Path(results_path)
    # 1) Load documents (per page)
    # docs = load_pdf_as_documents(pdf_path, per_page=True)
    # print(f"Loaded {len(docs)} Document(s) from PDF.")

    segments = load_asr_ocr_segments_from_json(json_path)

    print(f"Loaded {len(segments)} segment Document(s) from JSON.")

    if not segments:
        print("No segments found. Check your JSON structure.")
        return []

    first = segments[0]
    print("\nFirst segment:")
    print("  text    :")
    print(first.text[:400], "...\n")
    
    


    # 2) Create core components
    chunker = ASRTimestampChunker(max_seconds=15.0, overlap_seconds=3.0)
    embedder = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
    dim = embedder.dimension
    store = FaissVectorStore(dim=dim)
    llm =llm = HuggingFaceLLM(
    model_name="Qwen/Qwen3-0.6B",
    device="auto",
    torch_dtype="auto",
    max_new_tokens=32768,
    temperature=0.7,
    top_p=0.95,
    is_chat_model=True,
    system_prompt="You are a helpful assistant that uses ONLY the provided video transcript context .",
)

    # 3) Build RAG pipeline
    rag = RAGPipeline(
        embedder=embedder,
        vector_store=store,
        llm=llm,
        chunker=chunker,
        default_top_k=5,
    )

    # 4) Index documents
    print("Indexing documents...")
    rag.add_segments(segments,document_id="lstm")

    print(f"Vector store size: {len(store)} chunks")

    # 5) Ask a question
    question = "what is forget gate do in LSTM?"
    print(f"\nQUESTION: {question}\n")

    result = rag.answer_question(question, top_k=50, return_sources=True)

    print("ANSWER:\n")
    print(result["answer"])
    # print("\nSOURCES:")
    # for s in result["sources"]:
    #     print(
    #         f"- chunk_id={s['chunk_id']}, doc={s['document_id']}, "
    #         f"score={s['score']:.3f}, metadata={s['metadata']}"
    #     )


if __name__ == "__main__":
    # Run interactive mode (default)
    main()
    
    # Or run direct mode with a specific video (comment main() above and uncomment below)
    # result = process_single_video_direct("video.mp4")  # Local file
    # result = process_single_video_direct("https://youtube.com/watch?v=ID")  # URL
    
    # Or run quick URL test (uncomment below)
    # quick_url_test()