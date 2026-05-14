"""Load catalog JSON and provide allowlisted lookups."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Iterable

from app.config import Settings, get_settings
from app.models.catalog import CatalogAssessment

_lock = Lock()
_catalog: list[CatalogAssessment] | None = None
_by_url: dict[str, CatalogAssessment] | None = None


def _normalize_url(url: str) -> str:
    return url.strip()


def load_catalog(path: Path | None = None) -> list[CatalogAssessment]:
    global _catalog, _by_url
    settings = get_settings()
    p = path or settings.catalog_path
    with _lock:
        if _catalog is not None and path is None:
            return _catalog
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        items = [CatalogAssessment.model_validate(x) for x in raw]
        _catalog = items
        _by_url = {_normalize_url(i.url): i for i in items}
        return _catalog


def get_by_url(url: str) -> CatalogAssessment | None:
    if _by_url is None:
        load_catalog()
    assert _by_url is not None
    return _by_url.get(_normalize_url(url))


def allowlisted_urls() -> set[str]:
    if _by_url is None:
        load_catalog()
    return set(_by_url.keys()) if _by_url else set()


def find_best_name_match(name: str, candidates: Iterable[CatalogAssessment]) -> CatalogAssessment | None:
    name_l = name.strip().lower()
    best: CatalogAssessment | None = None
    best_score = 0.55
    for c in candidates:
        cname = c.name.strip().lower()
        if name_l == cname:
            return c
        score = SequenceMatcher(None, name_l, cname).ratio()
        if name_l in cname or cname in name_l:
            score = max(score, 0.72)
        if score > best_score:
            best_score = score
            best = c
    return best


def find_mentions_in_text(text: str, catalog: list[CatalogAssessment]) -> list[CatalogAssessment]:
    """Heuristic: detect catalog product names mentioned in free text."""
    t = text.lower()
    hits: list[CatalogAssessment] = []
    for item in catalog:
        n = item.name.lower()
        if len(n) < 4:
            continue
        if re.search(rf"\b{re.escape(n)}\b", t):
            hits.append(item)
            continue
        # allow partial for very long names
        if len(n) > 20 and n in t:
            hits.append(item)
    return hits
