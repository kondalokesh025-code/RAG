import repository_tree 
print( repository_tree.__file__)

import ingestion
print("INGESTION MODULE:", ingestion.__file__, flush=True)
from repository_tree import build_repository_inventory, save_repository_inventory

from giturlparse import parse
from collections import Counter
import time
from aider_repomap import build_aider_repo_map
from ingestion import CACHE_ROOT, prepare_repository, ingest_snapshot
from chunking import chunk_documents
from embedding import get_embedding_model
from vector_store import create_vector_store
from retrieval import build_retrievers
from eval_retrieval import evaluate_retrieval
from structural_analysis import extract_file_structure
from cache import (
    build_processing_fingerprint,
    build_index_fingerprint,
    load_processed_artifacts,
    save_processed_artifacts
)


url = "https://github.com/pallets/itsdangerous.git"
parsed = parse(url)
owner, repo = parsed.owner, parsed.repo
ref = None

EXPECTED_FILES = [
    "src/itsdangerous/__init__.py",
    "src/itsdangerous/encoding.py",
    "src/itsdangerous/exc.py",
    "src/itsdangerous/serializer.py",
    "src/itsdangerous/signer.py",
    "src/itsdangerous/timed.py",
    "src/itsdangerous/url_safe.py",
]


    
if __name__ == "__main__":
    total_start = time.perf_counter()

    print("\n===== REPOSITORY SNAPSHOT =====")
    snapshot = prepare_repository(owner, repo, ref=ref)
    commit_sha = snapshot["commit_sha"]
    print("Commit:", commit_sha)
    print("Snapshot cache:", "HIT" if snapshot["cache_hit"] else "MISS")

    inventory = build_repository_inventory(snapshot)
    tree_dir = CACHE_ROOT / "analysis" / owner.lower() / repo.lower() / commit_sha
    paths = save_repository_inventory(inventory, tree_dir)

    print("\n===== REPOSITORY TREE =====")
    print(inventory["tree"])
    print("Inventory:", paths)
    print("Summary:", inventory["summary"])



    print("\n===== AIDER REPOSITORY MAP =====")
    start = time.perf_counter()

    repo_map_result = build_aider_repo_map(
        snapshot,
        tree_dir,
        map_tokens=2048,
        verbose=True,
    )

    print("Tracked files:", repo_map_result.source_file_count)
    print("Estimated tokens:", repo_map_result.estimated_tokens)
    print("Repository map:", repo_map_result.markdown_path)
    print("Map metadata:", repo_map_result.json_path)
    print(f"RepoMap time: {time.perf_counter() - start:.2f}s")

    processing_fingerprint = build_processing_fingerprint(owner, repo, commit_sha)

    print("\n===== PROCESSED ARTIFACT CACHE =====")
    cache_start = time.perf_counter()
    cached = load_processed_artifacts(owner, repo, commit_sha, processing_fingerprint, CACHE_ROOT)
    cache_load_time = time.perf_counter() - cache_start

    if cached is not None:
        documents = cached["documents"]
        structure_records = cached["structure_records"]
        chunked_documents = cached["chunks"]
        ingestion_time = structural_time = chunking_time = None
        print("Processed cache: HIT")

    else:
        print("Processed cache: MISS")

        print("BEFORE INGESTION", flush=True)
        start = time.perf_counter()

        try:
            documents = ingestion.ingest_snapshot(snapshot)
        except Exception as exc:
            print("INGESTION ERROR:", repr(exc), flush=True)
            raise

        ingestion_time = time.perf_counter() - start
        print("AFTER INGESTION:", len(documents), flush=True)

        print("BEFORE STRUCTURAL ANALYSIS", flush=True)

        start = time.perf_counter()
        code_docs = [doc for doc in documents if doc.metadata.get("category") == "code"]
        structure_records = []

        for i, doc in enumerate(code_docs, 1):
            source = doc.metadata.get("source", "unknown")
            print(f"PROCESSING {i}/{len(code_docs)}: {source}", flush=True)

            try:
                record = extract_file_structure(doc)
            except BaseException as exc:
                print(f"FAILED: {source} | {type(exc).__name__}: {exc}", flush=True)
                raise

            structure_records.append(record)
            print(f"DONE: {source} | Tags: {len(record.get('tags', []))}", flush=True)

        structural_time = time.perf_counter() - start
        print("AFTER STRUCTURAL ANALYSIS:", len(structure_records), flush=True)

        print("BEFORE CHUNKING", flush=True)
        start = time.perf_counter()
        chunked_documents = chunk_documents(documents)
        chunking_time = time.perf_counter() - start
        print("AFTER CHUNKING:", len(chunked_documents), flush=True)

        if not chunked_documents:
            raise RuntimeError("No chunks were created.")

        start = time.perf_counter()
        save_processed_artifacts(owner, repo, commit_sha, processing_fingerprint, CACHE_ROOT, documents, structure_records, chunked_documents)
        print(f"Processed artifacts saved in {time.perf_counter() - start:.2f}s")

    print(f"Cache load time: {cache_load_time:.2f}s")

    if not chunked_documents:
        raise RuntimeError("No chunks were created.")

    print("\n===== INGESTION =====")
    document_paths = {doc.metadata.get("source", "") for doc in documents}
    print("Documents:", len(documents))
    print("Unique files:", len(document_paths))

    print("\nSource-file coverage after ingestion:")
    for path in EXPECTED_FILES:
        print(f"{'FOUND' if path in document_paths else 'MISSING'}: {path}")

    print("Ingestion time:", f"{ingestion_time:.2f}s" if ingestion_time is not None else "skipped (processed cache hit)")

    print("\n===== STRUCTURAL ANALYSIS =====")
    status_counts = Counter(record["status"] for record in structure_records)
    tag_counts = Counter(record["tag_status"] for record in structure_records)

    print("Analysis statuses:", dict(status_counts))
    print("Tag statuses:", dict(tag_counts))
    print("Definitions:", sum(len(r["definitions"]) for r in structure_records))
    print("References:", sum(len(r["references"]) for r in structure_records))
    print("Structural time:", f"{structural_time:.2f}s" if structural_time is not None else "skipped (processed cache hit)")

    print("\n===== CHUNKING =====")
    chunk_paths = {chunk.metadata.get("source", "") for chunk in chunked_documents}
    print("Total chunks:", len(chunked_documents))
    print("Unique files represented:", len(chunk_paths))

    print("\nSource-file coverage after chunking:")
    for path in EXPECTED_FILES:
        print(f"{'FOUND' if path in chunk_paths else 'MISSING'}: {path}")

    print("\nChunks by category:")
    print(Counter(chunk.metadata.get("category", "unknown") for chunk in chunked_documents))
    print("Chunking time:", f"{chunking_time:.2f}s" if chunking_time is not None else "skipped (processed cache hit)")

    print("\n===== EMBEDDING MODEL =====")
    start = time.perf_counter()
    embed_model = get_embedding_model()
    print(f"Model loading time: {time.perf_counter() - start:.2f}s")

    print("\n===== VECTOR STORE =====")
    index_fingerprint = build_index_fingerprint(processing_fingerprint)
    start = time.perf_counter()
    vector_store = create_vector_store(chunked_documents, embed_model, owner, repo, commit_sha=commit_sha, fingerprint=index_fingerprint)
    print(f"Indexing time: {time.perf_counter() - start:.2f}s")

    print("\n===== RETRIEVAL EVALUATION =====")
    start = time.perf_counter()
    rerank = build_retrievers(chunked_documents, vector_store)
    print(f"Retriever setup time: {time.perf_counter() - start:.2f}s")
    evaluate_retrieval(chunked_documents, vector_store, rerank)

    print(f"\nTotal runtime: {time.perf_counter() - total_start:.2f}s")
    print("Generation skipped.")
