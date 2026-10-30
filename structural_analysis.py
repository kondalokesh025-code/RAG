
from functools import lru_cache

from tree_sitter import Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language, get_tags_query

from chunking import TREE_SITTER_LANGUAGES
from language_utils import normalize_language


MAX_SOURCE_BYTES = 1_000_000
SCHEMA_VERSION = 3

PARSER_ALIASES = {"shell": "bash"}
MAX_DIAGNOSTICS = 50


@lru_cache(maxsize=None)
def get_language_object(language):
    return get_language(language)


@lru_cache(maxsize=None)
def get_tags_query_object(language):
    query_source = get_tags_query(language)
    if not query_source:
        return None
    return Query(get_language_object(language), query_source)


def extract_tags(content, parser_language, source):
    source_bytes = content.encode("utf-8")
    parser = Parser(get_language_object(parser_language))
    tree = parser.parse(source_bytes)
    query = get_tags_query_object(parser_language)

    if query is None:
        return [], "query_unavailable", tree

    cursor = QueryCursor(query)
    matches = cursor.matches(tree.root_node)

    tags = []
    seen = set()

    for _, match in matches:
        name_nodes = match.get("name", [])

        for capture_name, kind_nodes in match.items():
            if capture_name.startswith("definition."):
                role, kind = "definition", capture_name.split(".", 1)[1]
            elif capture_name.startswith("reference."):
                role, kind = "reference", capture_name.split(".", 1)[1]
            else:
                continue

            for kind_node in kind_nodes:
                for node in name_nodes:
                    if not (kind_node.start_byte <= node.start_byte and node.end_byte <= kind_node.end_byte):
                        continue

                    name = source_bytes[node.start_byte:node.end_byte].decode("utf-8").strip()
                    if not name:
                        continue

                    key = (role, kind, node.start_byte, node.end_byte)
                    if key in seen:
                        continue
                    seen.add(key)

                    tags.append({
                        "id": f"{source}::{role}.{kind}@{node.start_point.row + 1}:{node.start_point.column + 1}",
                        "role": role,
                        "kind": kind,
                        "name": name,
                        "line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "column": node.start_point.column + 1,
                        "end_column": node.end_point.column + 1,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                    })

    tags.sort(key=lambda item: (item["line"], item["column"], item["role"]))

    if not tags:
        return tags, "no_matches", tree

    roles = {tag["role"] for tag in tags}
    status = "ok" if roles == {"definition", "reference"} else "definitions_only" if roles == {"definition"} else "references_only"

    return tags, status, tree


def extract_diagnostics(root):
    diagnostics = []
    total = 0
    stack = [root]

    while stack:
        node = stack.pop()

        if node.is_error or node.is_missing:
            total += 1

            if len(diagnostics) < MAX_DIAGNOSTICS:
                diagnostics.append({
                    "type": "missing_node" if node.is_missing else "syntax_error",
                    "node_type": node.type,
                    "line": node.start_point.row + 1,
                    "column": node.start_point.column + 1,
                })

        stack.extend(reversed(node.children))

    return diagnostics, total


def extract_file_structure(doc):
    metadata = doc.metadata
    content = doc.page_content
    source = metadata.get("source") or "<unknown>"
    language = normalize_language(metadata.get("language"))
    parser_language = PARSER_ALIASES.get(language, TREE_SITTER_LANGUAGES.get(language))

    record = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "language": language,
        "category": metadata.get("category", "unknown"),
        "cell_index": metadata.get("cell_index"),
        "location_scope": "cell" if metadata.get("cell_index") is not None else "file",
        "status": None,
        "tag_status": "not_run",
        "definitions": [],
        "references": [],
        "tags": [],
        "diagnostics": [],
    }

    if metadata.get("category") != "code":
        record["status"] = "skipped_non_code"
        record["tag_status"] = "skipped_non_code"
        return record

    if parser_language is None:
        record["status"] = "unsupported_language"
        record["tag_status"] = "unsupported_language"
        return record

    source_bytes = content.encode("utf-8")
    if len(source_bytes) > MAX_SOURCE_BYTES:
        record["status"] = "skipped_too_large"
        record["tag_status"] = "skipped_too_large"
        return record

    source_key = f"{source}#cell={metadata['cell_index']}" if metadata.get("cell_index") is not None else source

    try:
        tags, tag_status, tree = extract_tags(content, parser_language, source_key)
        diagnostics, diagnostic_count = extract_diagnostics(tree.root_node)

        record["tags"] = tags
        record["definitions"] = [tag for tag in tags if tag["role"] == "definition"]
        record["references"] = [tag for tag in tags if tag["role"] == "reference"]
        record["tag_status"] = tag_status
        record["diagnostics"] = diagnostics
        record["diagnostic_count"] = diagnostic_count
        record["diagnostics_truncated"] = diagnostic_count > MAX_DIAGNOSTICS
        record["status"] = "ok_with_diagnostics" if diagnostic_count else "ok"

    except Exception as exc:
        record["status"] = "parse_failed"
        record["tag_status"] = "failed"
        record["error"] = str(exc)

    return record
