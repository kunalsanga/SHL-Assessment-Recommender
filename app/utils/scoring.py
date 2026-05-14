"""Hybrid scoring: semantic similarity plus lexical overlap."""

from __future__ import annotations

from app.models.catalog import CatalogAssessment
from app.utils.text import tokenize_for_overlap


def keyword_overlap_score(query_text: str, item: CatalogAssessment) -> float:
    q_tokens = tokenize_for_overlap(query_text)
    if not q_tokens:
        return 0.0
    blob = " ".join(
        [
            item.name,
            item.description,
            " ".join(item.skills),
            " ".join(item.job_roles),
            " ".join(item.test_type_labels),
        ]
    )
    doc_tokens = tokenize_for_overlap(blob)
    if not doc_tokens:
        return 0.0
    inter = len(q_tokens & doc_tokens)
    union = len(q_tokens | doc_tokens)
    return inter / union if union else 0.0


def skill_overlap_score(skills: list[str], item: CatalogAssessment) -> float:
    if not skills:
        return 0.0
    want = tokenize_for_overlap(" ".join(skills))
    have = tokenize_for_overlap(" ".join(item.skills) + " " + item.name + " " + item.description)
    if not want or not have:
        return 0.0
    inter = len(want & have)
    return inter / len(want)


def role_relevance_score(roles_hint: str, item: CatalogAssessment) -> float:
    if not roles_hint.strip():
        return 0.0
    q = tokenize_for_overlap(roles_hint)
    jr = tokenize_for_overlap(" ".join(item.job_roles))
    if not q or not jr:
        return 0.0
    inter = len(q & jr)
    return inter / max(len(q), 1)


def type_preference_score(preferences: list[str], item: CatalogAssessment) -> float:
    """
    Map user preference phrases to SHL test type codes (A–S).
    Returns [0,1] based on overlap between inferred codes and item codes.
    """
    if not preferences:
        return 0.0
    blob = " ".join(preferences).lower()
    inferred: set[str] = set()
    mapping = [
        (["personality", "behavior", "behaviour", "opq", "traits"], ["P"]),
        (["cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal"], ["A"]),
        (["situational", "judgement", "judgment", "biodata", "sjt"], ["B"]),
        (["competenc"], ["C"]),
        (["360", "development"], ["D"]),
        (["exercise", "assessment center", "assessment centre"], ["E"]),
        (["skill", "technical", "coding", "java", "python", "knowledge"], ["K"]),
        (["simulation", "work sample", "inbox"], ["S"]),
    ]
    for keys, codes in mapping:
        if any(k in blob for k in keys):
            inferred.update(codes)
    if not inferred:
        return 0.0
    item_codes = {c.upper() for c in item.test_type_codes}
    if not item_codes:
        return 0.0
    hit = len(inferred & item_codes)
    return hit / len(inferred)
