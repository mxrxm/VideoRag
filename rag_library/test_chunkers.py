from pathlib import Path

# --- Loaders ---
from loaders import load_pdf_as_documents, load_pptx_as_documents,load_asr_ocr_segments_from_json
# --- Chunkers ---
from chuncking import SimpleTextSplitter, SentenceTextSplitter, SemanticChunker ,ASRTimestampChunker
# --- Embeddings / Vectorstore / LLM / Pipeline ---
from embeddings import HuggingFaceEmbedding
from vectordb import FaissVectorStore , ChromaVectorStore
from llm import HuggingFaceLLM
from pipeline import RAGPipeline


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



def main():
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
        path = input("\nEnter Video JSON path: ").strip().strip('"')
        doc_id = input("Enter document_id (default=video): ").strip() or "video"
        run_video_json_flow(rag, path, document_id=doc_id)

    print(f"\nVector store size: {len(store)} chunks")
    interactive_qa(rag)


if __name__ == "__main__":
    main()
