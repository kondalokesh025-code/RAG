from langchain_chroma import Chroma


import hashlib


def get_collection_name(owner, repo):
    repository_id = f"{owner}/{repo}".lower()
    repo_hash = hashlib.sha256(
        repository_id.encode("utf-8")
    ).hexdigest()[:12]

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






def create_vector_store(chunked_documents, embed_model, owner, repo):
    collection_name = get_collection_name(owner, repo)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embed_model,
        persist_directory="./chroma_db"
    )

    vector_store.delete_collection()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embed_model,
        persist_directory="./chroma_db"
    )

    if chunked_documents:
        chunk_ids = generate_chunk_ids(
            chunked_documents, owner, repo
        )

        vector_store.add_documents(
            documents=chunked_documents,
            ids=chunk_ids
        )

    return vector_store


