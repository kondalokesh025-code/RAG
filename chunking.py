
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownTextSplitter
from langchain_core.documents import Document
from tree_sitter_language_pack import process, ProcessConfig


CHUNKER_REGISTRY = {
    ("code", "python"): Language.PYTHON,
    ("code", "cpp"): Language.CPP,
    ("code", "c"): Language.C,
    ("code", "go"): Language.GO,
    ("code", "java"): Language.JAVA,
    ("code", "kotlin"): Language.KOTLIN,
    ("code", "javascript"): Language.JS,
    ("code", "typescript"): Language.TS,
    ("code", "php"): Language.PHP,
    ("code", "protobuf"): Language.PROTO,
    ("code", "r"): Language.R,
    ("code", "ruby"): Language.RUBY,
    ("code", "rust"): Language.RUST,
    ("code", "scala"): Language.SCALA,
    ("code", "swift"): Language.SWIFT,
    ("code", "lua"): Language.LUA,
    ("code", "perl"): Language.PERL,
    ("code", "haskell"): Language.HASKELL,
    ("code", "elixir"): Language.ELIXIR,
    ("code", "powershell"): Language.POWERSHELL,
    ("markup", "markdown"): Language.MARKDOWN,
    ("markup", "rst"): Language.RST,
    ("markup", "html"): Language.HTML,
}


TREE_SITTER_LANGUAGES = {
    "python": "python", "cpp": "cpp", "c": "c", "go": "go",
    "java": "java", "kotlin": "kotlin", "javascript": "javascript",
    "typescript": "typescript", "php": "php", "protobuf": "proto",
    "r": "r", "ruby": "ruby", "rust": "rust", "scala": "scala",
    "swift": "swift", "lua": "lua", "perl": "perl", "haskell": "haskell",
    "elixir": "elixir", "powershell": "powershell", "bash": "bash",
    "json": "json", "yaml": "yaml", "sql": "sql", "html": "html",
    "css": "css", "dockerfile": "dockerfile", "toml": "toml",
    "xml": "xml", "make": "make", "markdown": "markdown",
    "julia": "julia", "dart": "dart", "zig": "zig",
    "fsharp": "fsharp", "ocaml": "ocaml",
}


CATEGORY_MAP = {
    "code": None,
    "markup": None,
    "config": None,
    "notebook": None,
}


def tree_sitter_chunk(doc):
    language = doc.metadata.get("language", "").lower()
    parser_language = TREE_SITTER_LANGUAGES.get(language)

    if parser_language is None:
        return []

    try:
        result = process(doc.page_content, ProcessConfig(language=parser_language, chunk_max_size=1000))
    except Exception:
        return []

    chunked_documents = []

    for chunk in result.chunks:
        metadata = {
            **doc.metadata,
            "chunker": "tree_sitter",
            "start_line": chunk.start_line + 1,
            "end_line": chunk.end_line,
            "chunk_index": chunk.metadata.chunk_index,
        }
        chunked_documents.append(Document(page_content=chunk.content, metadata=metadata))

    return chunked_documents


def chunk_documents(documents):
    chunked_documents = []

    for doc in documents:
        category = doc.metadata.get("category")
        language = (doc.metadata.get("language") or "").lower()
        splitter_language = CHUNKER_REGISTRY.get((category, language))

        if category == "code" and language in TREE_SITTER_LANGUAGES:
            ast_chunks = tree_sitter_chunk(doc)
            if ast_chunks:
                chunked_documents.extend(ast_chunks)
                continue

        if splitter_language is not None:
            splitter = RecursiveCharacterTextSplitter.from_language(language=splitter_language, chunk_size=500, chunk_overlap=100)
        elif CATEGORY_MAP.get(category) == Language.MARKDOWN:
            splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        chunked_documents.extend(splitter.split_documents([doc]))

    return chunked_documents
