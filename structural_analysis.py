from tree_sitter_language_pack import process, ProcessConfig, get_parser

from chunking import TREE_SITTER_LANGUAGES
from language_utils import normalize_language


MAX_SOURCE_BYTES = 1_000_000
SCHEMA_VERSION = 1

PARSER_ALIASES = {"shell": "bash"}


def get_line_range(span):
    start_line = span.start_line + 1
    end_line = span.end_line if span.end_column == 0 and span.end_line > span.start_line else span.end_line + 1
    return start_line, max(start_line, end_line)


def get_value(value):
    return str(getattr(value, "value", value))


def walk_nodes(node):
    yield node
    for child in node.children:
        yield from walk_nodes(child)


def get_node_text(node, source_bytes):
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def get_python_syntax_root(content):
    tree = get_parser("python").parse(content.encode("utf-8"))
    return tree.root_node if tree is not None else None


def get_decorator_starts(root, source_bytes):
    starts = {}

    if root is None:
        return starts

    for node in walk_nodes(root):
        if node.type not in {"function_definition", "class_definition"}:
            continue

        parent = node.parent

        if parent is None or parent.type != "decorated_definition":
            continue

        name_node = node.child_by_field_name("name")

        if name_node is None:
            continue

        name = get_node_text(name_node, source_bytes)
        definition_line = node.start_point.row + 1
        decorator_line = parent.start_point.row + 1
        starts[(name, definition_line)] = decorator_line

    return starts


def is_module_assignment(node):
    parent = node.parent

    while parent is not None and parent.type == "expression_statement":
        parent = parent.parent

    return parent is not None and parent.type == "module"



def get_string_literal_value(node, source_bytes):
    raw = get_node_text(node, source_bytes)
    prefix_end = 0

    while prefix_end < len(raw) and raw[prefix_end].lower() in "rubf":
        prefix_end += 1

    prefix = raw[:prefix_end].lower()
    literal = raw[prefix_end:]

    if "b" in prefix or "f" in prefix:
        return None

    if literal.startswith(("'''", '"""')):
        delimiter = literal[:3]
    elif literal.startswith(("'", '"')):
        delimiter = literal[0]
    else:
        return None

    if not literal.endswith(delimiter):
        return None

    value = literal[len(delimiter):-len(delimiter)]

    if "\\" in value and "r" not in prefix:
        return None

    return value


def get_python_all_exports(root, source_bytes, source_key):
    if root is None:
        return []

    export_names = []
    export_node = None

    for node in walk_nodes(root):
        if node.type not in {"assignment", "annotated_assignment"}:
            continue

        if not is_module_assignment(node):
            continue

        left = node.child_by_field_name("left")
        value = node.child_by_field_name("right")

        if value is None:
            value = node.child_by_field_name("value")

        if left is None or get_node_text(left, source_bytes).strip() != "__all__":
            continue

        export_node = node
        export_names = []

        if value is None or value.type not in {"list", "tuple", "set"}:
            continue

        for child in value.named_children:
            if child.type != "string":
                continue

            name = get_string_literal_value(child, source_bytes)

            if name is not None:
                export_names.append(name)

    if not export_names or export_node is None:
        return []

    start_line = export_node.start_point.row + 1
    end_line = export_node.end_point.row + 1

    return [
        {
            "id": f"{source_key}::export@{start_line}:{name}",
            "name": name,
            "kind": "declared_public_export",
            "start_line": start_line,
            "end_line": end_line,
        }
        for name in dict.fromkeys(export_names)
    ]

def extract_file_structure(doc):
    metadata = doc.metadata
    content = doc.page_content
    source = metadata.get("source") or "<unknown>"
    language = normalize_language(metadata.get("language"))
    cell_index = metadata.get("cell_index")
    source_key = f"{source}#cell={cell_index}" if cell_index is not None else source

    record = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "language": language,
        "cell_index": cell_index,
        "location_scope": "cell" if cell_index is not None else "file",
        "status": None,
        "definitions": [],
        "imports": [],
        "exports": [],
        "docstrings": [],
        "diagnostics": [],
    }

    if metadata.get("category") != "code":
        record["status"] = "skipped_non_code"
        return record

    parser_language = PARSER_ALIASES.get(language, TREE_SITTER_LANGUAGES.get(language))

    if parser_language is None:
        record["status"] = "unsupported_language"
        return record

    if len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        record["status"] = "skipped_too_large"
        return record

    try:
        parsed = process(content, ProcessConfig(language=parser_language, structure=True, imports=True, exports=True, docstrings=True, diagnostics=True, chunk_max_size=None))
    except Exception as exc:
        record["status"] = "parse_failed"
        record["error"] = str(exc)
        return record

    source_bytes = content.encode("utf-8")
    raw_root = None
    decorator_starts = {}

    if language == "python":
        try:
            raw_root = get_python_syntax_root(content)
            decorator_starts = get_decorator_starts(raw_root, source_bytes)
        except Exception as exc:
            record["diagnostics"].append({"message": f"Raw syntax inspection failed: {exc}"})

    def extract_definitions(items, parent=None):
        for item in items:
            start_line, end_line = get_line_range(item.span)
            name = item.name or f"<anonymous@{start_line}>"
            qualified_name = f"{parent}.{name}" if parent else name

            if language == "python":
                start_line = decorator_starts.get((name, start_line), start_line)

            record["definitions"].append({
                "id": f"{source_key}::{qualified_name}@{start_line}",
                "name": item.name,
                "qualified_name": qualified_name,
                "parent": parent,
                "kind": get_value(item.kind),
                "signature": item.signature,
                "doc_comment": item.doc_comment,
                "visibility": item.visibility,
                "start_line": start_line,
                "end_line": end_line,
            })

            extract_definitions(item.children, qualified_name)

    extract_definitions(parsed.structure)

    for index, item in enumerate(parsed.imports):
        start_line, end_line = get_line_range(item.span)
        record["imports"].append({
            "id": f"{source_key}::import@{start_line}:{index}",
            "statement": item.source,
            "items": list(item.items),
            "alias": item.alias,
            "is_wildcard": item.is_wildcard,
            "start_line": start_line,
            "end_line": end_line,
        })

    for index, item in enumerate(parsed.exports):
        start_line, end_line = get_line_range(item.span)
        record["exports"].append({
            "id": f"{source_key}::export@{start_line}:{index}",
            "name": item.name,
            "kind": get_value(item.kind),
            "start_line": start_line,
            "end_line": end_line,
        })

    if language == "python":
        existing_exports = {item["name"] for item in record["exports"] if item["name"]}
        for item in get_python_all_exports(raw_root, source_bytes, source_key):
            if item["name"] not in existing_exports:
                record["exports"].append(item)
                existing_exports.add(item["name"])

    for index, item in enumerate(parsed.docstrings):
        start_line, end_line = get_line_range(item.span)
        record["docstrings"].append({
            "id": f"{source_key}::docstring@{start_line}:{index}",
            "associated_item": item.associated_item,
            "format": get_value(item.format),
            "text": item.text,
            "start_line": start_line,
            "end_line": end_line,
        })

    for item in parsed.diagnostics:
        start_line, end_line = get_line_range(item.span)
        record["diagnostics"].append({
            "message": item.message,
            "severity": get_value(item.severity),
            "start_line": start_line,
            "end_line": end_line,
        })

    record["status"] = "ok_with_diagnostics" if record["diagnostics"] else "ok"
    return record