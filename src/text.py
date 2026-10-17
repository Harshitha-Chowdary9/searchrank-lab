"""Shared tokenization for the whole project (BM25, features, API)."""
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alphanumeric tokenization. ``None``-safe."""
    return _TOKEN_RE.findall((text or "").lower())


def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))
