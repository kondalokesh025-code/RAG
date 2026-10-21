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


def normalize_language(language):
    if not isinstance(language, str) or not language.strip():
        return None

    language = language.strip().lower()
    return ALIASES.get(language, language)