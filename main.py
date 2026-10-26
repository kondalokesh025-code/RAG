from giturlparse import parse
from collections import Counter
import time
from ingestion import ingest_repository
from chunking import chunk_documents
from embedding import get_embedding_model
from vector_store import create_vector_store
from retrieval import build_retrievers
from eval_retrieval import evaluate_retrieval


url = "https://github.com/pallets/click.git"

parsed = parse(url)
owner = parsed.owner
repo = parsed.repo

EXPECTED_FILES = [
    "src/click/core.py",
    "src/click/decorators.py",
    "src/click/exceptions.py",
    "src/click/types.py",
    "src/click/testing.py",
    "src/click/parser.py",
    "src/click/utils.py",
    "src/click/termui.py",
]


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 1. INGESTION
    print("\n===== INGESTION =====")

    start = time.perf_counter()
    documents = ingest_repository(owner, repo)

    document_paths = {
        doc.metadata.get("source", "")
        for doc in documents
    }

    print("Documents:", len(documents))
    print("Unique files:", len(document_paths))

    print("\nSource-file coverage after ingestion:")
    for path in EXPECTED_FILES:
        status = "FOUND" if path in document_paths else "MISSING"
        print(f"{status}: {path}")

    print(f"Ingestion time: {time.perf_counter() - start:.2f}s")

    # 2. CHUNKING
    print("\n===== CHUNKING =====")

    start = time.perf_counter()
    chunked_documents = chunk_documents(documents)

    chunk_paths = {
        chunk.metadata.get("source", "")
        for chunk in chunked_documents
    }

    print("Total chunks:", len(chunked_documents))
    print("Unique files represented:", len(chunk_paths))

    print("\nSource-file coverage after chunking:")
    for path in EXPECTED_FILES:
        status = "FOUND" if path in chunk_paths else "MISSING"
        print(f"{status}: {path}")

    print("\nChunks by category:")
    print(Counter(
        chunk.metadata.get("category", "unknown")
        for chunk in chunked_documents
    ))

    print(f"Chunking time: {time.perf_counter() - start:.2f}s")

    if not chunked_documents:
        raise RuntimeError("No chunks were created.")

    # 3. EMBEDDING MODEL
    print("\n===== EMBEDDING MODEL =====")

    start = time.perf_counter()
    embed_model = get_embedding_model()
    print(f"Model loading time: {time.perf_counter() - start:.2f}s")

    # 4. VECTOR STORE
    print("\n===== VECTOR STORE =====")

    start = time.perf_counter()
    vector_store = create_vector_store(
        chunked_documents,
        embed_model,
        owner,
        repo,
    )
    print(f"Indexing time: {time.perf_counter() - start:.2f}s")

    # 5. RETRIEVAL EVALUATION — 23 queries
    print("\n===== RETRIEVAL EVALUATION =====")

    start = time.perf_counter()
    rerank = build_retrievers(chunked_documents, vector_store)

    print(f"Retriever setup time: {time.perf_counter() - start:.2f}s")

    evaluate_retrieval(
        chunked_documents,
        vector_store,
        rerank,
    )

    print(
        f"\nTotal runtime: {time.perf_counter() - start_time:.2f}s"
    )
    print("Generation skipped.")















