ALIASES = {
    "python3": "python",
    "python 3": "python",
    "ipython": "python",
    "py": "python",
    "c++": "cpp",
    "c#": "csharp",
    "c-sharp": "csharp",
    "js": "javascript",
    "node.js": "javascript",
    "ts": "typescript",
    "bash": "shell",
    "sh": "shell",
    "shellscript": "shell",
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


def normalize_language(language):
    if not isinstance(language, str) or not language.strip():
        return None

    language = language.strip().lower()
    return ALIASES.get(language, language)