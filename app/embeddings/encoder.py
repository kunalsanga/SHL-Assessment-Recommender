"""Sentence-transformers wrapper for consistent encoding."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    with _lock:
        if _model is None:
            _model = SentenceTransformer(MODEL_NAME)
        return _model


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray[Any, Any]:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_query(text: str) -> np.ndarray[Any, Any]:
    return encode_texts([text])[0]
