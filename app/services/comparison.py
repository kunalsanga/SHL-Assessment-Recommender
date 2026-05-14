"""Grounded comparison using catalog metadata only."""

from __future__ import annotations

from app.models.catalog import CatalogAssessment
from app.prompts.templates import comparison_system, comparison_user_block
from app.services.catalog_loader import find_best_name_match
from app.services.gemini_client import generate_text
from app.services.recommendation import format_snippet


def compare_by_names(
    name_a: str,
    name_b: str,
    catalog: list[CatalogAssessment],
) -> tuple[str, list[CatalogAssessment]]:
    a = find_best_name_match(name_a, catalog)
    b = find_best_name_match(name_b, catalog)
    if a is None or b is None:
        missing = []
        if a is None:
            missing.append(name_a)
        if b is None:
            missing.append(name_b)
        msg = (
            "I could not confidently match these names to the SHL Individual Test Solutions catalog: "
            + ", ".join(missing)
            + ". Please paste the exact catalog names or URLs from shl.com."
        )
        return msg, [x for x in (a, b) if x is not None]

    facts_a = format_snippet(a, max_chars=900)
    facts_b = format_snippet(b, max_chars=900)
    user = comparison_user_block(a.name, facts_a, b.name, facts_b)
    reply = generate_text(comparison_system(), user)
    return reply.strip(), [a, b]
