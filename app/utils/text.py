"""Text helpers for tokenization and normalization."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_overlap(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 2."""
    text = normalize_text(text)
    return {t for t in re.split(r"[^a-z0-9+#.]+", text) if len(t) >= 2}


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
