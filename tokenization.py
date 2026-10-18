import re 

TOKEN_PATTERN = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*(?:(?:\.|::|->)[A-Za-z_$][A-Za-z0-9_$]*)*|\d+(?:\.\d+)?"
)

CAMEL_CASE_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+"
)


def tokenize_identifier(identifier: str) -> list[str]:
    tokens = []
    seen = set()

    def add(token):
        token = token.lower()
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

    add(identifier)

    parts = re.split(r"::|->|\.", identifier)

    for part in parts:
        add(part)

        for word in part.split("_"):
            add(word)
            for subword in CAMEL_CASE_PATTERN.findall(word):
                add(subword)

    return tokens


def bm25_preprocess(text: str) -> list[str]:
    tokens = []

    for token in TOKEN_PATTERN.findall(text):
        tokens.extend(tokenize_identifier(token))

    return tokens