import json
import os
import posixpath
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 2

CODE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".go", ".rs", ".c", ".h", ".cc",
    ".cpp", ".hh", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".scala", ".ex", ".exs", ".erl", ".hs", ".lua", ".sh",
    ".dart", ".vue", ".svelte", ".r", ".R", ".jl", ".pl", ".pm",
    ".sql", ".html", ".css", ".scss", ".sass", ".less",
}

DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".tex"}
MARKUP_EXTENSIONS = {".svg"}

CONFIG_EXTENSIONS = {
    ".toml", ".ini", ".cfg", ".conf", ".yaml", ".yml", ".json",
    ".xml", ".properties", ".env", ".lock",
}

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".arrow",
    ".feather", ".sqlite", ".db", ".pkl", ".pickle", ".npy", ".npz",
}

ENTRYPOINT_NAMES = {
    "__main__.py", "main.py", "app.py", "run.py", "server.py", "manage.py",
    "wsgi.py", "asgi.py", "cli.py", "index.js", "index.ts", "index.tsx",
    "main.js", "main.ts", "main.go", "main.rs", "program.cs", "main.c",
    "main.cpp", "app.js", "app.ts",
}

CONFIG_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "pipfile", "pipfile.lock", "package.json", "tsconfig.json", "jsconfig.json",
    "cargo.toml", "cargo.lock", "go.mod", "go.sum", "pom.xml",
    "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
    "composer.json", "gemfile", "gemfile.lock", "makefile", "cmakelists.txt",
    "dockerfile", ".gitignore", ".pre-commit-config.yaml", "tox.ini",
    "pytest.ini", "ruff.toml", ".ruff.toml",
}

DEFAULT_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", "target", "vendor", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", "coverage", "htmlcov",
}


def normalize_path(path):
    return posixpath.normpath(
        str(path).replace("\\", "/")
    ).lstrip("/")


def classify_path(relative_path):
    """Return a broad, display-oriented file category."""
    path = normalize_path(relative_path)
    name = posixpath.basename(path)
    lower_name = name.lower()
    lower_path = path.lower()
    suffix = Path(name).suffix.lower()

    if (
        "/tests/" in f"/{lower_path}/" or "/test/" in f"/{lower_path}/"
        or lower_name.startswith("test_") or lower_name.endswith("_test.py")
        or lower_name.endswith((".spec.js", ".test.js", ".spec.ts", ".test.ts"))
    ):
        return "tests"

    if (
        lower_name in {"readme", "changelog", "contributing", "license", "authors"}
        or suffix in DOC_EXTENSIONS
    ):
        return "docs"

    if lower_name in CONFIG_NAMES or suffix in CONFIG_EXTENSIONS:
        return "config"

    if suffix in DATA_EXTENSIONS or any(
        part in {"data", "datasets", "fixtures"}
        for part in lower_path.split("/")
    ):
        return "data"

    if suffix in MARKUP_EXTENSIONS:
        return "markup"

    if suffix in CODE_EXTENSIONS:
        return "code"

    if not suffix and lower_name in {
        "makefile", "dockerfile", "justfile", "procfile"
    }:
        return "config"

    return "other"


def _ingestion_policy():
    """Reuse the ingestion module's filtering policy where available."""
    try:
        from ingestion import should_skip_file, SKIP_DIRS
        return should_skip_file, set(SKIP_DIRS) | DEFAULT_SKIP_DIRS
    except (ImportError, AttributeError):
        return None, DEFAULT_SKIP_DIRS


def _entry_roles(relative_path, category):
    name = posixpath.basename(relative_path)
    lower_name = name.lower()
    roles = []

    if lower_name in ENTRYPOINT_NAMES or lower_name.startswith("main."):
        roles.append("entry-point candidate")

    if (
        lower_name in CONFIG_NAMES
        or lower_name.startswith(("requirements", "dockerfile"))
    ):
        roles.append("project config")

    if category == "tests":
        roles.append("test")

    if category == "docs":
        roles.append("documentation")

    return roles


def _default_skip_reason(path, relative_path, size, category):
    if path.is_symlink():
        return "symlink"

    if category == "other":
        return "unsupported or unclassified file type"

    return None


def build_repository_inventory(snapshot):
    """Scan a prepare_repository() snapshot and return its inventory and tree."""
    root = Path(snapshot["repo_dir"]).resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            f"Repository snapshot does not exist: {root}"
        )

    owner = snapshot.get("owner", "")
    repo = snapshot.get("repo", root.name)
    commit_sha = snapshot.get("commit_sha", "")

    should_skip_file, skip_dirs = _ingestion_policy()

    entries = []
    directory_paths = set()

    for current, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        kept_dirs = []

        for dirname in sorted(dirnames):
            full_path = current_path / dirname
            relative = normalize_path(
                full_path.relative_to(root).as_posix()
            )

            directory_paths.add(relative)

            is_symlink = full_path.is_symlink()
            excluded = dirname in skip_dirs or is_symlink

            entries.append({
                "path": relative,
                "kind": "directory",
                "category": "directory",
                "size_bytes": None,
                "indexable": not excluded,
                "skip_reason": (
                    "symlink directory" if is_symlink
                    else "excluded directory" if excluded
                    else None
                ),
                "roles": [],
            })

            if not excluded:
                kept_dirs.append(dirname)

        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = current_path / filename
            relative = normalize_path(
                path.relative_to(root).as_posix()
            )
            category = classify_path(relative)

            try:
                size = path.stat(follow_symlinks=False).st_size
            except OSError as exc:
                entries.append({
                    "path": relative,
                    "kind": "file",
                    "category": category,
                    "size_bytes": None,
                    "indexable": False,
                    "skip_reason": f"stat failed: {exc}",
                    "roles": _entry_roles(relative, category),
                })
                continue

            reason = None

            if should_skip_file is not None:
                try:
                    reason = should_skip_file(path, relative, size)
                except Exception as exc:
                    reason = f"filter check failed: {exc}"

            if not reason:
                reason = _default_skip_reason(
                    path, relative, size, category
                )

            entries.append({
                "path": relative,
                "kind": "file",
                "category": category,
                "size_bytes": size,
                "indexable": reason is None,
                "skip_reason": str(reason) if reason else None,
                "roles": _entry_roles(relative, category),
            })

    entries.sort(
        key=lambda item: (item["path"].lower(), item["kind"])
    )

    tree = render_repository_tree(
        entries, f"{owner}/{repo}".strip("/")
    )

    file_entries = [
        item for item in entries
        if item["kind"] == "file"
    ]

    category_counts = Counter(
        item["category"] for item in file_entries
    )

    indexable_count = sum(
        item["indexable"] for item in file_entries
    )

    skipped_files = [
        item for item in file_entries
        if not item["indexable"]
    ]

    skip_counts = Counter(
        item["skip_reason"] for item in skipped_files
    )

    excluded_directory_count = sum(
        item["kind"] == "directory" and not item["indexable"]
        for item in entries
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "repository": f"{owner}/{repo}".strip("/"),
        "commit_sha": commit_sha,
        "root": str(root),
        "entries": entries,
        "tree": tree,
        "summary": {
            "file_count": len(file_entries),
            "directory_count": len(directory_paths),
            "excluded_directory_count": excluded_directory_count,
            "indexable_file_count": indexable_count,
            "skipped_file_count": len(skipped_files),
            "category_counts": dict(sorted(category_counts.items())),
            "skip_reason_counts": dict(skip_counts),

            "entry_point_candidates": [
                item["path"] for item in file_entries
                if "entry-point candidate" in item["roles"]
            ],

            "config_files": [
                item["path"] for item in file_entries
                if "project config" in item["roles"]
            ],
        },
    }


def render_repository_tree(entries, root_name="repository"):
    """Render the repository as an annotated ASCII tree."""
    tree_root = {"children": {}, "entry": None}

    for entry in entries:
        parts = normalize_path(entry["path"]).split("/")
        node = tree_root

        for part in parts:
            node = node["children"].setdefault(
                part, {"children": {}, "entry": None}
            )

        node["entry"] = entry

    lines = [f"{root_name or 'repository'}/"]

    def label_for(name, node):
        entry = node["entry"]

        is_dir = bool(node["children"]) or (
            entry and entry["kind"] == "directory"
        )

        label = name + ("/" if is_dir else "")

        if entry:
            roles = entry.get("roles", [])

            if roles:
                label += f" [{', '.join(roles)}]"
            elif entry["kind"] == "file":
                label += f" [{entry.get('category', 'other')}]"

            if not entry.get("indexable", True):
                label += (
                    f" [skipped: {entry.get('skip_reason') or 'excluded'}]"
                )

        return label

    def visit(node, prefix=""):
        def is_directory(pair):
            child = pair[1]
            entry = child["entry"]

            return bool(child["children"]) or (
                entry and entry["kind"] == "directory"
            )

        children = sorted(
            node["children"].items(),
            key=lambda pair: (
                not is_directory(pair),
                pair[0].lower()
            )
        )

        for index, (name, child) in enumerate(children):
            last = index == len(children) - 1

            lines.append(
                f"{prefix}{'└── ' if last else '├── '}"
                f"{label_for(name, child)}"
            )

            if child["children"]:
                visit(
                    child,
                    prefix + ("    " if last else "│   ")
                )

    visit(tree_root)

    return "\n".join(lines)


def save_repository_inventory(inventory, output_dir):
    """Save JSON inventory and readable Markdown tree."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "repository_inventory.json"
    tree_path = output_dir / "repository_tree.md"

    json_path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    summary = inventory["summary"]

    lines = [
        f"# Repository tree: {inventory['repository']}",
        "",
        f"- Commit: `{inventory['commit_sha']}`",
        f"- Files: {summary['file_count']}",
        f"- Indexable files: {summary['indexable_file_count']}",
        f"- Skipped files: {summary['skipped_file_count']}",
        f"- Directories: {summary['directory_count']}",
        f"- Excluded directories: {summary['excluded_directory_count']}",
        "",
        "## Entry-point candidates",
        *(
            [f"- `{path}`" for path in summary["entry_point_candidates"]]
            or ["- None detected by filename heuristics"]
        ),
        "",
        "## Project configuration",
        *(
            [f"- `{path}`" for path in summary["config_files"]]
            or ["- None detected by filename heuristics"]
        ),
        "",
        "## File tree",
        "",
        "```text",
        inventory["tree"],
        "```",
        "",
        "## Skipped file counts",
    ]

    if summary["skip_reason_counts"]:
        lines.extend(
            f"- {reason}: {count}"
            for reason, count in sorted(
                summary["skip_reason_counts"].items()
            )
        )
    else:
        lines.append("- No files skipped by the inventory filter")

    tree_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return {
        "json": str(json_path),
        "markdown": str(tree_path)
    }