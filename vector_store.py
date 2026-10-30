from langchain_chroma import Chroma
import hashlib


def get_collection_name(owner, repo, commit_sha, fingerprint="v1"):
    repository_id = f"{owner}/{repo}".lower()
    unique_key = f"{repository_id}/{commit_sha}/{fingerprint}"
    repo_hash = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:20]
    return f"repo_{repo_hash}"


def generate_chunk_ids(chunked_documents, owner, repo):
    chunk_ids = []
    source_counts = {}

    for doc in chunked_documents:
        source = doc.metadata["source"]
        chunk_index = source_counts.get(source, 0)
        source_counts[source] = chunk_index + 1

        unique_key = (
            f"{owner}/{repo}/{source}/{chunk_index}/"
            f"{doc.page_content}"
        )

        chunk_id = hashlib.sha256(
            unique_key.encode("utf-8")
        ).hexdigest()

        chunk_ids.append(chunk_id)

    return chunk_ids

def create_vector_store(
    chunked_documents, embed_model, owner, repo,
    commit_sha=None, fingerprint="v1", force_rebuild=False
):
    if not commit_sha:
        raise ValueError("commit_sha is required for revision-aware caching.")

    collection_name = get_collection_name(owner, repo, commit_sha, fingerprint)
    persist_directory = "./chroma_db"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embed_model,
        persist_directory=persist_directory
    )

    chunk_ids = generate_chunk_ids(chunked_documents, owner, repo)

    if force_rebuild:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embed_model,
            persist_directory=persist_directory
        )
        existing_ids = []
    else:
        existing_ids = vector_store._collection.get(include=[])["ids"]

        if existing_ids and set(existing_ids) != set(chunk_ids):
            vector_store.delete_collection()
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embed_model,
                persist_directory=persist_directory
            )
            existing_ids = []

    if not existing_ids and chunk_ids:
        vector_store.add_documents(documents=chunked_documents, ids=chunk_ids)
        print("Vector index: built")
    else:
        print("Vector index: cache hit")

    return vector_store