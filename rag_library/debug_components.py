from loaders import load_pdf_as_documents
from chuncking import SemanticChunker
from embeddings import HuggingFaceEmbedding
from vectordb import InMemoryVectorStore,FaissVectorStore
from retriever import SimpleRetriever
from pipeline import RAGPipeline


PDF_PATH = r"C:\Users\Fatma PC\Desktop\raglib\1706.03762v7.pdf"


def debug_loader():
    print("=== DEBUG: PDF Loader ===")
    docs = load_pdf_as_documents(PDF_PATH, per_page=True)
    print(f"Total docs (pages with text): {len(docs)}")
    first = docs[0]
    print(f"\nDoc[0].id = {first.id}")
    print(f"Doc[0].metadata = {first.metadata}")
    print("\n--- Doc[0] first 700 chars ---")
    print(first.text[:700])
    print("\n---------------------------")
    return docs


def debug_semantic_chunker(docs):
    print("\n=== DEBUG: SemanticChunker on first doc ===")

    embedder = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.70,
        max_sentences_per_chunk=6,
        min_sentences_per_chunk=1,
    )

    doc = docs[1]
    chunks = chunker.split_document(doc)
    print(f"Chunks from doc[0]: {len(chunks)}")

    for i, c in enumerate(chunks[:5]):
        print(f"\n[CHUNK {i}] id={c.id}")
        print(c.text[:400])
        print("metadata:", c.metadata)
    return embedder, chunks

def debug_vectorstore_roundtrip(embedder, chunks):
    print("\n=== DEBUG: VectorStore Roundtrip ===")


    # 1) embed chunks
    texts = [c.text for c in chunks]
    embs = embedder.encode(texts)
    for c, e in zip(chunks, embs):
        c.embedding = e

    
    store = FaissVectorStore(dim=embedder.dimension)
    store.add_chunks(chunks)
    print(f"Store size: {len(store)} chunks")

    # 2) Pick a chunk and query with its own text
    target = chunks[0]
    print("\nQuerying with CHUNK[0] text:")
    print(target.text[:200], "...\n")

    q_emb = embedder.encode_one(target.text)
    results = store.similarity_search_by_vector(q_emb, k=3)

    for r in results:
        print(f"score={r.score:.3f} | id={r.chunk.id}")
        print(r.chunk.text[:200], "...\n")

def debug_retrieval_full(docs):
    print("\n=== DEBUG: Retrieval for real question ===")
    embedder = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.70,
        max_sentences_per_chunk=6,
        min_sentences_per_chunk=1,
    )

    store = FaissVectorStore(dim=embedder.dimension)

    # indexing
    all_chunks = []
    for doc in docs:
        ch = chunker.split_document(doc)
        all_chunks.extend(ch)

    texts = [c.text for c in all_chunks]
    embs = embedder.encode(texts)
    for c, e in zip(all_chunks, embs):
        c.embedding = e
    store.add_chunks(all_chunks)
    print(f"Indexed chunks: {len(store)}")

    from retriever import SimpleRetriever
    retriever = SimpleRetriever(embedder=embedder, vector_store=store)

    question = "What is the main idea of this paper?"
    print(f"\nQUESTION: {question}\n")

    qwr = retriever.retrieve(question, k=50)

    for i, r in enumerate(qwr.results):
        print(f"[RESULT {i}] score={r.score:.3f} | id={r.chunk.id}")
        print(r.chunk.text[:500])
        print("metadata:", r.chunk.metadata)
        print("-" * 60)

if __name__ == "__main__":
    docs = debug_loader()
    # embedder, chunks = debug_semantic_chunker(docs)
    # debug_vectorstore_roundtrip(embedder, chunks)
    debug_retrieval_full(docs)
