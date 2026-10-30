import tempfile , json , os , hashlib 
from pathlib import Path
from langchain_core.documents import Document


CACHE_SCHEMA_VERSION = 1
PROCESSING_MODULES = ("ingestion.py", "chunking.py", "structural_analysis.py")
INDEX_MODULES = ("embedding.py", "vector_store.py")
FILTER_ENV_VARS = (
    "REPO_SKIP_DIRS", "REPO_SKIP_PATTERNS",
    "REPO_BINARY_EXTENSIONS", "REPO_SKIP_EXTENSIONS", "MAX_FILE_SIZE"
)
MODEL_ENV_VARS = (
    "EMBEDDING_MODEL", "EMBEDDING_MODEL_NAME",
    "HF_EMBEDDING_MODEL", "MODEL_NAME"
)
PROJECT_ROOT = Path(__file__).resolve().parent


def _file_hash(filename):
    path = PROJECT_ROOT / filename
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_data(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_processing_fingerprint(owner, repo, commit_sha):
    data = {
        "schema": CACHE_SCHEMA_VERSION,
        "repository": f"{owner}/{repo}".lower(),
        "commit_sha": commit_sha,
        "modules": {name: _file_hash(name) for name in PROCESSING_MODULES},
        "filter_settings": {name: os.getenv(name) for name in FILTER_ENV_VARS}
    }
    return _hash_data(data)


def build_index_fingerprint(processing_fingerprint):
    data = {
        "schema": CACHE_SCHEMA_VERSION,
        "processing_fingerprint": processing_fingerprint,
        "modules": {name: _file_hash(name) for name in INDEX_MODULES},
        "model_settings": {name: os.getenv(name) for name in MODEL_ENV_VARS}
    }
    return _hash_data(data)


def _cache_path(owner, repo, commit_sha, fingerprint, cache_root):
    return (
        Path(cache_root) / "processed" / owner.lower() / repo.lower()
        / commit_sha / f"{fingerprint}.json"
    )


def _serialize_documents(documents):
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]


def _restore_documents(items):
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in items
    ]


def load_processed_artifacts(
    owner, repo, commit_sha, fingerprint, cache_root
):
    path = _cache_path(owner, repo, commit_sha, fingerprint, cache_root)
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None

        if (
            data.get("schema") != CACHE_SCHEMA_VERSION
            or data.get("commit_sha") != commit_sha
            or data.get("fingerprint") != fingerprint
            or not all(
                isinstance(data.get(key), list)
                for key in ("documents", "structure_records", "chunks")
            )
        ):
            return None

        return {
            "documents": _restore_documents(data["documents"]),
            "structure_records": data["structure_records"],
            "chunks": _restore_documents(data["chunks"])
        }

    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Processed cache could not be loaded: {exc}")
        return None


def save_processed_artifacts(
    owner, repo, commit_sha, fingerprint, cache_root,
    documents, structure_records, chunks
):
    path = _cache_path(owner, repo, commit_sha, fingerprint, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema": CACHE_SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "fingerprint": fingerprint,
        "documents": _serialize_documents(documents),
        "structure_records": structure_records,
        "chunks": _serialize_documents(chunks)
    }

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            suffix=".tmp", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            json.dump(data, temp, ensure_ascii=False, separators=(",", ":"))

        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
