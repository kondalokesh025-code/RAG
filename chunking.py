from langchain_text_splitters import RecursiveCharacterTextSplitter , Language , MarkdownTextSplitter
from  giturlparse import parse 
from ingestion import ingest_repository





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
    ("code", "solidity"): Language.SOL,
    ("code", "csharp"): Language.CSHARP,
    ("code", "lua"): Language.LUA,
    ("code", "perl"): Language.PERL,
    ("code", "haskell"): Language.HASKELL,
    ("code", "elixir"): Language.ELIXIR,
    ("code", "powershell"): Language.POWERSHELL,
    ("markup", "markdown"): Language.MARKDOWN,
    ("markup", "rst"): Language.RST,
    ("markup", "html"): Language.HTML,
}


CATEGORY_MAP = {
    "code": None,
    "markup": None , 
    "config": None,
    "notebook": None,
}

def chunk_documents(documents) : 
    chunked_documents = []


    for doc in documents:
        category = doc.metadata.get("category")
        language = doc.metadata.get("language")

        splitter_language = CHUNKER_REGISTRY.get((category, language))

        if splitter_language is not None:
            splitter = RecursiveCharacterTextSplitter.from_language(language=splitter_language,chunk_size=500,chunk_overlap=100)

        elif CATEGORY_MAP.get(category) == Language.MARKDOWN:
            splitter = MarkdownTextSplitter(chunk_size=500,chunk_overlap=100)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        chunked_documents.extend(splitter.split_documents([doc]))
    return chunked_documents





