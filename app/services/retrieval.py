"""FAISS-backed hybrid retrieval (semantic + lexical)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import get_settings
from app.embeddings.encoder import encode_query
from app.models.catalog import CatalogAssessment
from app.services.catalog_loader import load_catalog
from app.utils.scoring import keyword_overlap_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedItem:
    assessment: CatalogAssessment
    semantic_score: float
    keyword_score: float


class HybridRetriever:
    def __init__(
        self,
        catalog: list[CatalogAssessment],
        index: faiss.Index,
        url_by_row: list[str],
    ) -> None:
        self._catalog = catalog
        self._index = index
        self._url_by_row = url_by_row
        self._by_url = {a.url: a for a in catalog}

    @classmethod
    def from_disk(cls) -> HybridRetriever:
        settings = get_settings()
        catalog = load_catalog()
        meta_path = Path(settings.catalog_meta_path)
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing {meta_path}. Run scripts/build_embeddings.py and scripts/build_faiss_index.py"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        urls = meta["urls"]
        index = faiss.read_index(str(settings.faiss_index_path))
        if index.ntotal != len(urls):
            raise ValueError("FAISS index size does not match catalog_meta urls length")
        if len(catalog) != len(urls):
            logger.warning(
                "Catalog count (%s) != meta urls (%s). Alignment may be wrong; rebuild index.",
                len(catalog),
                len(urls),
            )
        return cls(catalog=catalog, index=index, url_by_row=urls)

    def retrieve(self, query_text: str, top_k: int) -> list[RetrievedItem]:
        if not query_text.strip():
            return []
        q = encode_query(query_text).astype("float32", copy=False)
        q = q.reshape(1, -1)
        scores, idxs = self._index.search(q, top_k)
        row_scores = scores[0]
        row_idxs = idxs[0]
        out: list[RetrievedItem] = []
        for row, s in zip(row_idxs, row_scores, strict=False):
            if row < 0:
                continue
            url = self._url_by_row[row]
            item = self._by_url.get(url)
            if item is None:
                continue
            kw = keyword_overlap_score(query_text, item)
            out.append(RetrievedItem(assessment=item, semantic_score=float(s), keyword_score=float(kw)))
        return out
