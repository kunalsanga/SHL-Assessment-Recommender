"""Ranking and building recommendation payloads from catalog only."""

from __future__ import annotations

import re
from typing import Iterable

from app.config import get_settings
from app.models.catalog import CatalogAssessment
from app.models.schemas import RecommendationItem
from app.services.retrieval import RetrievedItem
from app.utils.scoring import role_relevance_score, skill_overlap_score, type_preference_score


def _primary_test_type_label(item: CatalogAssessment) -> str:
    if item.test_type_labels:
        return item.test_type_labels[0]
    if item.test_type_codes:
        return item.test_type_codes[0]
    return "Unknown"


def user_wants_personality(last_user: str) -> bool:
    return bool(re.search(r"personality|opq|traits|behavior|behaviour", last_user, re.I))


def rank_items(
    retrieved: list[RetrievedItem],
    state_query: str,
    technical_skills: list[str],
    assessment_preferences: list[str],
    last_user: str,
) -> list[tuple[float, CatalogAssessment]]:
    settings = get_settings()
    scored: list[tuple[float, CatalogAssessment]] = []
    personality_boost = 0.08 if user_wants_personality(last_user) else 0.0
    for r in retrieved:
        a = r.assessment
        hybrid = 0.62 * r.semantic_score + 0.38 * r.keyword_score
        skill = skill_overlap_score(technical_skills, a)
        role = role_relevance_score(state_query, a)
        pref = type_preference_score(assessment_preferences, a)
        final = hybrid + 0.18 * skill + 0.12 * role + 0.10 * pref
        if personality_boost and "P" in {c.upper() for c in a.test_type_codes}:
            final += personality_boost
        scored.append((final, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    deduped: list[tuple[float, CatalogAssessment]] = []
    seen: set[str] = set()
    for score, a in scored:
        if a.url in seen:
            continue
        seen.add(a.url)
        deduped.append((score, a))
    return deduped[: max(settings.final_recommendation_max, settings.retrieval_top_k)]


def select_recommendation_count(confidence: float) -> int:
    """Map confidence to [1,10] count; caller ensures enough items exist."""
    if confidence >= 0.82:
        return 10
    if confidence >= 0.72:
        return 7
    if confidence >= 0.62:
        return 5
    if confidence >= 0.52:
        return 3
    return 1


def compute_confidence(
    has_role: bool,
    has_skills: bool,
    has_prefs: bool,
    top_rank_score: float,
) -> float:
    base = 0.25
    if has_role:
        base += 0.18
    if has_skills:
        base += 0.22
    if has_prefs:
        base += 0.12
    base += max(0.0, min(0.45, top_rank_score))
    return min(1.0, base)


def build_recommendation_items(assessments: Iterable[CatalogAssessment]) -> list[RecommendationItem]:
    out: list[RecommendationItem] = []
    for a in assessments:
        out.append(
            RecommendationItem(
                name=a.name,
                url=a.url,
                test_type=_primary_test_type_label(a),
            )
        )
    return out


def format_snippet(item: CatalogAssessment, max_chars: int = 420) -> str:
    type_blob = ", ".join(item.test_type_labels or item.test_type_codes)
    parts = [
        f"Name: {item.name}",
        f"URL: {item.url}",
        f"Test types: {type_blob}",
        f"Duration (min): {item.duration_minutes}",
        f"Remote testing: {item.remote_testing_supported}",
        f"Job levels: {', '.join(item.job_roles)}",
        f"Skills/topics: {', '.join(item.skills[:12])}",
        f"Description: {item.description}",
    ]
    text = "\n".join(parts)
    return text[:max_chars]
