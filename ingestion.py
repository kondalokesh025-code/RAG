import base64, fnmatch, json, os, re, shutil, subprocess, tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from language_utils import normalize_language






FILE_MAP = {
    # Programming languages
    ".py": ("code", "python"),
    ".pyw": ("code", "python"),
    ".pyi": ("code", "python"),
    ".c": ("code", "c"),
    ".h": ("code", "c"),
    ".cpp": ("code", "cpp"),
    ".cc": ("code", "cpp"),
    ".cxx": ("code", "cpp"),
    ".hpp": ("code", "cpp"),
    ".hh": ("code", "cpp"),
    ".go": ("code", "go"),
    ".rs": ("code", "rust"),
    ".rb": ("code", "ruby"),
    ".php": ("code", "php"),
    ".cs": ("code", "csharp"),
    ".swift": ("code", "swift"),
    ".kt": ("code", "kotlin"),
    ".kts": ("code", "kotlin"),
    ".java": ("code", "java"),
    ".js": ("code", "javascript"),
    ".jsx": ("code", "javascript"),
    ".mjs": ("code", "javascript"),
    ".cjs": ("code", "javascript"),
    ".ts": ("code", "typescript"),
    ".tsx": ("code", "typescript"),
    ".scala": ("code", "scala"),
    ".sc": ("code", "scala"),
    ".dart": ("code", "dart"),
    ".r": ("code", "r"),
    ".lua": ("code", "lua"),
    ".pl": ("code", "perl"),
    ".pm": ("code", "perl"),
    ".sh": ("code", "shell"),
    ".bash": ("code", "shell"),
    ".zsh": ("code", "shell"),
    ".fish": ("code", "fish"),
    ".ps1": ("code", "powershell"),
    ".bat": ("code", "batch"),
    ".cmd": ("code", "batch"),
    ".sql": ("code", "sql"),
    ".ex": ("code", "elixir"),
    ".exs": ("code", "elixir"),
    ".erl": ("code", "erlang"),
    ".hrl": ("code", "erlang"),
    ".hs": ("code", "haskell"),
    ".lhs": ("code", "haskell"),
    ".clj": ("code", "clojure"),
    ".cljs": ("code", "clojure"),
    ".cljc": ("code", "clojure"),
    ".elm": ("code", "elm"),
    ".f": ("code", "fortran"),
    ".for": ("code", "fortran"),
    ".f90": ("code", "fortran"),
    ".jl": ("code", "julia"),
    ".ml": ("code", "ocaml"),
    ".mli": ("code", "ocaml"),
    ".plsql": ("code", "sql"),
    ".sol": ("code", "solidity"),
    ".vue": ("code", "vue"),
    ".svelte": ("code", "svelte"),
    ".gradle": ("code", "groovy"),
    ".groovy": ("code", "groovy"),
    ".v": ("code", "verilog"),
    ".sv": ("code", "systemverilog"),
    ".zig": ("code", "zig"),
    ".nim": ("code", "nim"),
    ".vhd": ("code", "vhdl"),
    ".vhdl": ("code", "vhdl"),
    ".css": ("code", "css"),
    ".scss": ("code", "scss"),
    ".sass": ("code", "sass"),
    ".less": ("code", "less"),
    ".graphql": ("code", "graphql"),
    ".gql": ("code", "graphql"),
    ".proto": ("code", "protobuf"),
    ".nix": ("code", "nix"),

    # Markup and documentation
    ".md": ("markup", "markdown"),
    ".mdx": ("markup", "markdown"),
    ".rst": ("markup", "rst"),
    ".html": ("markup", "html"),
    ".htm": ("markup", "html"),
    ".xml": ("markup", "xml"),
    ".svg": ("markup", "xml"),

    # Notebooks
    ".ipynb": ("notebook", None),

    # Configuration and data formats
    ".json": ("config", "json"),
    ".jsonc": ("config", "json"),
    ".json5": ("config", "json"),
    ".yaml": ("config", "yaml"),
    ".yml": ("config", "yaml"),
    ".toml": ("config", "toml"),
    ".ini": ("config", "ini"),
    ".cfg": ("config", "ini"),
    ".conf": ("config", "text"),
    ".properties": ("config", "properties"),
    ".env": ("config", "text"),
    ".txt": ("config", "text"),
    ".csv": ("config", "csv"),
    ".tsv": ("config", "text"),
    ".tf": ("config", "hcl"),
    ".tfvars": ("config", "hcl"),
    ".hcl": ("config", "hcl"),
    ".make": ("config", "makefile"),
    ".mk": ("config", "makefile"),

    # Common extensionless filenames
    "readme": ("markup", "markdown"),
    "license": ("markup", "text"),
    "copying": ("markup", "text"),
    "notice": ("markup", "text"),
    "changelog": ("markup", "markdown"),
    "contributing": ("markup", "markdown"),
    "dockerfile": ("config", "dockerfile"),
    "containerfile": ("config", "dockerfile"),
    "makefile": ("config", "makefile"),
    "gemfile": ("config", "ruby"),
    "rakefile": ("config", "ruby"),
    "procfile": ("config", "text"),
    "justfile": ("config", "text"),
    "taskfile": ("config", "yaml"),
    ".gitignore": ("config", "text"),
    ".gitattributes": ("config", "text"),
    ".dockerignore": ("config", "text"),
    "cargo.lock": ("config", "text"),
    "poetry.lock": ("config", "text"),
    "yarn.lock": ("config", "text"),
    "gemfile.lock": ("config", "text"),
    "uv.lock": ("config", "text"),
    "go.sum": ("config", "text"),
    "mix.lock": ("config", "elixir"),
    "bun.lock": ("config", "text"),
    "flake.lock": ("config", "json"),
}



GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CACHE_ROOT = Path(os.getenv("REPO_CACHE_DIR", ".repo_cache")).resolve()

try:
    MAX_FILE_SIZE = max(1, int(os.getenv("MAX_FILE_SIZE_BYTES", "1000000")))
except ValueError:
    MAX_FILE_SIZE = 1_000_000


def env_list(name):
    value = os.getenv(name)
    return None if value is None else [x.strip() for x in value.split(",") if x.strip()]


DEFAULT_SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "site-packages", ".tox", ".mypy_cache", ".pytest_cache", ".next", ".nuxt", ".turbo", "coverage", "vendor", "third_party", "dist", "build", "target"}
SKIP_DIRS = {x.lower() for x in (env_list("REPO_SKIP_DIRS") or DEFAULT_SKIP_DIRS)} | {".git"}

SKIP_PATTERNS = env_list("REPO_SKIP_PATTERNS") or ["*.min.js", "*.min.css", "*.map", "*_pb2.py", "*.generated.*", "*_generated.*"]

BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyo", ".class", ".jar", ".war", ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth", ".mp3", ".wav", ".mp4", ".mov", ".avi", ".woff", ".woff2", ".ttf", ".otf", ".parquet", ".feather", ".arrow"}

if env_list("REPO_BINARY_EXTENSIONS") is not None:
    BINARY_EXTENSIONS = {x.lower() if x.startswith(".") else f".{x.lower()}" for x in env_list("REPO_BINARY_EXTENSIONS")}

BINARY_EXTENSIONS.update(x.lower() if x.startswith(".") else f".{x.lower()}" for x in (env_list("REPO_SKIP_EXTENSIONS") or []))


def run_git(args, cwd=None):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    if GITHUB_TOKEN:
        i = int(env.get("GIT_CONFIG_COUNT", "0"))
        token = base64.b64encode(f"x-access-token:{GITHUB_TOKEN}".encode()).decode()
        env.update({"GIT_CONFIG_COUNT": str(i + 1), f"GIT_CONFIG_KEY_{i}": "http.https://github.com/.extraheader", f"GIT_CONFIG_VALUE_{i}": f"AUTHORIZATION: basic {token}"})

    try:
        result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as e:
        raise RuntimeError("Git is not installed or not available in PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Git command timed out.") from e

    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed.")

    return result.stdout.strip()


def validate_ref(ref):
    if ref is None:
        return None
    if not isinstance(ref, str) or not ref or ref != ref.strip() or len(ref) > 255:
        raise ValueError("Invalid repository ref.")
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower()
    if ref.startswith("-") or ref in {".", "..", "@"} or any(c.isspace() or c in "~^:?*[\\" for c in ref) or ".." in ref or "@{" in ref or "//" in ref or ref.endswith(("/", ".")):
        raise ValueError("Invalid repository ref.")
    if any(not p or p.startswith(".") or p.endswith(".") or p.endswith(".lock") for p in ref.split("/")):
        raise ValueError("Invalid repository ref.")
    return ref


def resolve_commit(repo_url, ref=None):
    ref = validate_ref(ref)

    if ref and re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower(), ref.lower()

    if ref is None:
        output = run_git(["ls-remote", "--symref", repo_url, "HEAD"])
        lines = [line.split() for line in output.splitlines()]
        branch = next((p[1] for p in lines if len(p) == 3 and p[0] == "ref:" and p[2] == "HEAD"), "HEAD")
        sha = next((p[0] for p in lines if len(p) == 2 and p[1] == "HEAD"), None)
        if not sha:
            raise RuntimeError("Could not resolve the repository's default branch.")
        return sha, branch

    if ref.startswith("refs/heads/"):
        patterns, candidates = [ref], [ref]
    elif ref.startswith("refs/tags/"):
        patterns, candidates = [ref, f"{ref}^{{}}"], [f"{ref}^{{}}", ref]
    else:
        patterns = [f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"]
        candidates = [patterns[0], patterns[2], patterns[1]]

    refs = {}
    for line in run_git(["ls-remote", repo_url, *patterns]).splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs[parts[1]] = parts[0]

    for candidate in candidates:
        if candidate in refs:
            return refs[candidate], ref

    raise RuntimeError(f"Could not resolve repository ref: {ref}")


def snapshot_is_valid(path, commit_sha):
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        return run_git(["rev-parse", "--verify", "HEAD"], path) == commit_sha and not run_git(["status", "--porcelain", "--untracked-files=all"], path)
    except RuntimeError:
        return False


def prepare_repository(owner, repo, ref=None, cache_root=None):
    if not isinstance(owner, str) or not isinstance(repo, str):
        raise ValueError("Owner and repository must be strings.")

    owner, repo = owner.strip(), repo.strip().removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", owner) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", repo) or repo.endswith("."):
        raise ValueError("Invalid GitHub owner or repository.")

    repo_url = f"https://github.com/{owner}/{repo}.git"
    sha, resolved_ref = resolve_commit(repo_url, ref)
    identity = f"{owner.lower()}/{repo.lower()}"
    parent = (Path(cache_root).expanduser().resolve() if cache_root else CACHE_ROOT) / owner.lower() / repo.lower()
    target = parent / sha
    parent.mkdir(parents=True, exist_ok=True)

    if snapshot_is_valid(target, sha):
        return {"owner": owner, "repo": repo, "repo_identity": identity, "repo_dir": str(target), "commit_sha": sha, "ref": resolved_ref, "cache_hit": True}

    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)

    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=parent))
    try:
        run_git(["init", "--quiet"], staging)
        run_git(["remote", "add", "origin", repo_url], staging)
        run_git(["fetch", "--depth", "1", "origin", sha], staging)
        run_git(["checkout", "--detach", "FETCH_HEAD"], staging)
        if run_git(["rev-parse", "--verify", "HEAD"], staging) != sha:
            raise RuntimeError("Checked-out commit does not match the resolved commit.")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"owner": owner, "repo": repo, "repo_identity": identity, "repo_dir": str(target), "commit_sha": sha, "ref": resolved_ref, "cache_hit": False}


def classify_file(file):
    path = Path(file["path"])
    category, language = FILE_MAP.get(path.name.lower(), FILE_MAP.get(path.suffix.lower(), ("unknown", None)))
    return {"path": file["path"], "category": category, "language": language}


def should_skip_file(path, relative_path, size):
    if path.is_symlink():
        return "symlink"
    if size > MAX_FILE_SIZE:
        return "too_large"
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return "binary"
    if any(fnmatch.fnmatchcase(relative_path.lower(), p.lower()) or fnmatch.fnmatchcase(path.name.lower(), p.lower()) for p in SKIP_PATTERNS):
        return "generated"
    if classify_file({"path": relative_path})["category"] == "unknown":
        return "unsupported"
    try:
        with path.open("rb") as f:
            if b"\x00" in f.read(4096):
                return "binary"
    except OSError:
        return "unreadable"
    return None


def scan_repository_files(repo_dir):
    root = Path(repo_dir).resolve()
    files, stats = [], {"files_seen": 0, "files_selected": 0, "files_skipped": 0, "directories_skipped": 0, "skip_reasons": {}}

    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept = [d for d in dirs if not (current_path / d).is_symlink() and d.lower() not in SKIP_DIRS]
        stats["directories_skipped"] += len(dirs) - len(kept)
        dirs[:] = kept

        for name in names:
            path = current_path / name
            stats["files_seen"] += 1
            relative = path.relative_to(root).as_posix()
            try:
                reason = should_skip_file(path, relative, path.lstat().st_size)
            except OSError:
                reason = "unreadable"

            if reason:
                stats["files_skipped"] += 1
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                continue

            info = classify_file({"path": relative})
            info["size"] = path.lstat().st_size
            files.append(info)
            stats["files_selected"] += 1

    return files, stats


def process_file(file, raw_content):
    try:
        content = raw_content.decode("utf-8")
        return json.loads(content) if file["category"] == "notebook" else content
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def ingest_snapshot(snapshot, return_stats=False):
    files, stats = scan_repository_files(snapshot["repo_dir"])
    documents = []
    stats.update({"read_failures": 0, "parse_failures": 0})
    repo_dir = Path(snapshot["repo_dir"])
    base = {"repo": snapshot["repo_identity"], "commit_sha": snapshot["commit_sha"]}

    for file in files:
        try:
            content = process_file(file, (repo_dir / file["path"]).read_bytes())
        except OSError:
            stats["read_failures"] += 1
            continue

        if content is None:
            stats["parse_failures"] += 1
            continue

        metadata = {**base, "source": file["path"]}

        if isinstance(content, dict):
            nb = content.get("metadata") or {}
            info, kernel = nb.get("language_info") or {}, nb.get("kernelspec") or {}
            language = normalize_language(info.get("name") or kernel.get("language")) or "python"

            for i, cell in enumerate(content.get("cells", [])):
                kind, source = cell.get("cell_type"), cell.get("source", "")
                source = "".join(source) if isinstance(source, list) else source
                if not isinstance(source, str) or not source.strip():
                    continue
                if kind not in {"code", "markdown"}:
                    continue
                documents.append(Document(page_content=source, metadata={**metadata, "category": "code" if kind == "code" else "markup", "language": language if kind == "code" else "markdown", "cell_type": kind, "cell_index": i}))

        elif isinstance(content, str):
            documents.append(Document(page_content=content, metadata={**metadata, "category": file["category"], "language": file["language"] or "text"}))

    stats["documents"] = len(documents)
    return (documents, stats) if return_stats else documents


def ingest_repository(owner, repo, ref=None, cache_root=None):
    snapshot = prepare_repository(owner, repo, ref, cache_root)
    documents, stats = ingest_snapshot(snapshot, return_stats=True)
    print(f"Commit: {snapshot['commit_sha']} | Snapshot cache: {'HIT' if snapshot['cache_hit'] else 'MISS'}")
    print(f"Files scanned: {stats['files_seen']} | Selected: {stats['files_selected']} | Skipped: {stats['files_skipped']} | Documents: {stats['documents']}")
    if stats["skip_reasons"]:
        print(f"Skip reasons: {stats['skip_reasons']}")
    if stats["read_failures"] or stats["parse_failures"]:
        print(f"Read failures: {stats['read_failures']} | Parse failures: {stats['parse_failures']}")
    return documents