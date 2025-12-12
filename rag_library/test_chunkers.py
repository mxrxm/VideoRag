# from Loaders import load_pdf_as_documents, load_pptx_as_documents
# from core.Models import Document
# from chuncking import (
#     SimpleTextSplitter,
#     SentenceTextSplitter,
#     SemanticChunker,
# )
# from embeddings import HuggingFaceEmbedding


# def test_pdf_with_chunkers():
#     print("\n====================")
#     print("PDF + Chunkers")
#     print("====================")

#     pdf_path = r"C:\Users\Fatma PC\Desktop\raglib\1706.03762v7.pdf"  # <-- change this

#     docs = load_pdf_as_documents(pdf_path, per_page=True)
#     print(f"Loaded {len(docs)} Document(s) from PDF.")

#     if not docs:
#         print("No text found in PDF.")
#         return

#     doc: Document = docs[0]
#     print(f"\nUsing first document: id={doc.id}, metadata={doc.metadata}")

#     simple = SimpleTextSplitter(max_chars=300, overlap=50)
#     simple_chunks = simple.split_document(doc)
#     print(f"\nSimpleTextSplitter produced {len(simple_chunks)} chunks.")
#     for c in simple_chunks[:3]:
#         print(f"[{c.id}] {c.text[:120]}...")
#         print("-" * 40)

#     sent = SentenceTextSplitter(max_sentences=3, overlap_sentences=1)
#     sent_chunks = sent.split_document(doc)
#     print(f"\nSentenceTextSplitter produced {len(sent_chunks)} chunks.")
#     for c in sent_chunks[:3]:
#         print(f"[{c.id}] {c.text[:120]}...")
#         print("-" * 40)

#     embedder = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
#     semantic = SemanticChunker(embedder=embedder, similarity_threshold=0.7)
#     sem_chunks = semantic.split_document(doc)
#     print(f"\nSemanticChunker produced {len(sem_chunks)} chunks.")
#     for c in sem_chunks[:3]:
#         print(f"[{c.id}] {c.text[:120]}...")
#         print("-" * 40)


# def test_pptx_with_chunkers():
#     print("\n====================")
#     print("PPTX + Chunkers")
#     print("====================")

#     pptx_path = r"C:\Users\Fatma PC\Desktop\raglib\SIST_L4_Open Redirects[1].pptx"  # <-- change this

#     docs = load_pptx_as_documents(pptx_path, per_slide=True)
#     print(f"Loaded {len(docs)} Document(s) from PPTX.")

#     if not docs:
#         print("No text found in PPTX.")
#         return

#     doc: Document = docs[0]
#     print(f"\nUsing first slide document: id={doc.id}, metadata={doc.metadata}")

#     simple = SimpleTextSplitter(max_chars=300, overlap=50)
#     simple_chunks = simple.split_document(doc)
#     print(f"\nSimpleTextSplitter produced {len(simple_chunks)} chunks.")
#     for c in simple_chunks[:3]:
#         print(f"[{c.id}] {c.text[:120]}...")
#         print("-" * 40)

#     sent = SentenceTextSplitter(max_sentences=3, overlap_sentences=1)
#     sent_chunks = sent.split_document(doc)
#     print(f"\nSentenceTextSplitter produced {len(sent_chunks)} chunks.")
#     for c in sent_chunks[:3]:
#         print(f"[{c.id}] {c.text[:120]}...")
#         print("-" * 40)

#     embedder = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
#     semantic = SemanticChunker(embedder=embedder, similarity_threshold=0.7)
#     sem_chunks = semantic.split_document(doc)
#     print(f"\nSemanticChunker produced {len(sem_chunks)} chunks.")
#     for c in sem_chunks[:3]:
#         print(f"[{c.id}] {c.text[:120]}...")
#         print("-" * 40)


# if __name__ == "__main__":
#     test_pdf_with_chunkers()
#     test_pptx_with_chunkers()


# from core.Models import Document
# from loaders import load_pdf_as_documents
# from chuncking import SimpleTextSplitter
# from embeddings import HuggingFaceEmbedding
# from vectordb import InMemoryVectorStore, FaissVectorStore, ChromaVectorStore


# def build_chunks_from_first_pdf_page(pdf_path: str):
#     docs = load_pdf_as_documents(pdf_path, per_page=True)
#     print(f"Loaded {len(docs)} PDF docs")
#     if not docs:
#         return []

#     doc: Document = docs[0]
#     splitter = SimpleTextSplitter(max_chars=500, overlap=100)
#     chunks = splitter.split_document(doc)
#     print(f"Chunked into {len(chunks)} chunks")
#     return chunks


# def embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     embedder = HuggingFaceEmbedding(model_name)
#     texts = [c.text for c in chunks]
#     embs = embedder.encode(texts)
#     for c, e in zip(chunks, embs):
#         c.embedding = e
#     return embedder, chunks


# def test_inmemory(pdf_path: str):
#     print("\n=== InMemoryVectorStore ===")
#     chunks = build_chunks_from_first_pdf_page(pdf_path)
#     if not chunks:
#         return
#     embedder, chunks = embed_chunks(chunks)

#     store = InMemoryVectorStore()
#     store.add_chunks(chunks)
#     print(f"Store size: {len(store)}")

#     q = "What is this paper about?"
#     q_emb = embedder.encode_one(q)
#     results = store.similarity_search_by_vector(q_emb, k=3)
#     for r in results:
#         print(f"Score={r.score:.3f}, chunk_id={r.chunk.id}")
#         print(r.chunk.text[:200], "...")
#         print("-" * 40)


# def test_faiss(pdf_path: str):
#     print("\n=== FaissVectorStore ===")
#     chunks = build_chunks_from_first_pdf_page(pdf_path)
#     if not chunks:
#         return
#     embedder, chunks = embed_chunks(chunks)

#     dim = embedder.dimension
#     store = FaissVectorStore(dim=dim)
#     store.add_chunks(chunks)
#     print(f"Store size: {len(store)}")

#     q = "What is this paper about?"
#     q_emb = embedder.encode_one(q)
#     results = store.similarity_search_by_vector(q_emb, k=3)
#     for r in results:
#         print(f"Score={r.score:.3f}, chunk_id={r.chunk.id}")
#         print(r.chunk.text[:200], "...")
#         print("-" * 40)


# def test_chroma(pdf_path: str):
#     print("\n=== ChromaVectorStore ===")
#     chunks = build_chunks_from_first_pdf_page(pdf_path)
#     if not chunks:
#         return
#     embedder, chunks = embed_chunks(chunks)

#     store = ChromaVectorStore(collection_name="raglib_test")
#     store.add_chunks(chunks)
#     print(f"Store size: {len(store)}")

#     q = "What is this paper about?"
#     q_emb = embedder.encode_one(q)
#     results = store.similarity_search_by_vector(q_emb, k=3)
#     for r in results:
#         print(f"Score={r.score:.3f}, chunk_id={r.chunk.id}")
#         print(r.chunk.text[:200], "...")
#         print("-" * 40)


# if __name__ == "__main__":
#     pdf_path = r"C:\Users\Fatma PC\Desktop\raglib\1706.03762v7.pdf"  # your PDF
#     test_inmemory(pdf_path)
#     test_faiss(pdf_path)
#     test_chroma(pdf_path)

from loaders import load_pdf_as_documents
from embeddings import HuggingFaceEmbedding
from vectordb import InMemoryVectorStore ,FaissVectorStore
from llm import HuggingFaceLLM
from pipeline import RAGPipeline
from chuncking import SemanticChunker,SimpleTextSplitter,ASRTimestampChunker
from core.utils import load_asr_ocr_segments_from_json


def main():
    pdf_path = r"C:\Users\Fatma PC\Desktop\raglib\1706.03762v7.pdf"  # your Transformer paper
    json_path=r"C:\Users\Fatma PC\Desktop\raglib\sample_ocr_test.json"
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
    main()


