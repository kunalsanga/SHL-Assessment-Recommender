"""Orchestrates extraction, retrieval, grounded replies, and strict catalog outputs."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse, RecommendationItem
from app.prompts.templates import REPLY_SYSTEM
from app.services.catalog_loader import allowlisted_urls, load_catalog
from app.services.comparison import compare_by_names
from app.services.gemini_client import generate_text
from app.services.refusal import refusal_reply, rules_refusal_reason
from app.services.retrieval import HybridRetriever
from app.services.recommendation import (
    build_recommendation_items,
    compute_confidence,
    format_snippet,
    rank_items,
    select_recommendation_count,
)
from app.services.state_manager import (
    ConversationState,
    StateExtraction,
    extract_state_and_intent,
    state_to_query_text,
)

logger = logging.getLogger(__name__)


def _messages_dicts(messages: list[Any]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _user_turn_count(messages: list[dict[str, str]]) -> int:
    return sum(1 for m in messages if m["role"] == "user")


def _split_compare_from_text(text: str) -> list[str]:
    parts = re.split(r"\s+(?:vs\.?|versus|compared to)\s+", text.strip(), flags=re.I)
    out = [p.strip(" ?.\"") for p in parts if p.strip()]
    return [p for p in out if len(p) >= 2][:3]


def _compare_names(extraction: StateExtraction, last_user: str) -> list[str]:
    if extraction.compare_names and len(extraction.compare_names) >= 2:
        return extraction.compare_names[:3]
    m = re.search(r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", last_user, re.I)
    if m:
        return [m.group(1).strip(" \""), m.group(2).strip(" \"")]
    if re.search(r"\bvs\.?|versus\b", last_user, re.I):
        parts = _split_compare_from_text(last_user)
        if len(parts) >= 2:
            return parts
    if extraction.intent == "compare":
        parts = _split_compare_from_text(last_user)
        if len(parts) >= 2:
            return parts
    return []


def _wants_compare(extraction: StateExtraction, last_user: str) -> bool:
    if extraction.intent == "compare":
        return True
    if re.search(r"\bdifference between\b", last_user, re.I):
        return True
    if re.search(r"\bvs\.?|versus\b", last_user, re.I):
        return True
    return False


def _other_intent_reply() -> str:
    return (
        "I only help with SHL Individual Test Solutions: shortlisting assessments, comparisons, "
        "and clarifying what to measure. Tell me the role, seniority, and whether you care about "
        "technical skills, personality, or cognitive ability."
    )


def _build_reply_prompt(
    *,
    state: ConversationState,
    last_user: str,
    snippets: list[str],
    allowed_names: list[str],
    recommendations_empty: bool,
) -> str:
    if recommendations_empty:
        return (
            f"User (latest): {last_user}\n\n"
            f"Structured state:\n{state.model_dump()}\n\n"
            "Recommendations list is EMPTY. Ask 1–2 focused follow-ups to gather missing context. "
            "Do not claim specific products yet.\n"
        )
    joined = "\n\n---\n\n".join(snippets)
    names_line = ", ".join(allowed_names)
    return (
        f"User (latest): {last_user}\n\n"
        f"Structured state:\n{state.model_dump()}\n\n"
        f"You may reference ONLY these catalog assessment names in your answer: {names_line}\n\n"
        f"Catalog snippets (ground truth):\n{joined}\n\n"
        "Explain briefly why each recommended assessment fits, using only snippet facts.\n"
    )


def _validate_recommendations(items: list[RecommendationItem]) -> list[RecommendationItem]:
    allow = allowlisted_urls()
    bad = [i for i in items if i.url not in allow]
    if bad:
        logger.error("Blocked non-catalog URL(s) from entering response: %s", bad)
        return [i for i in items if i.url in allow]
    return items


def process_chat(req: ChatRequest, retriever: HybridRetriever | None) -> ChatResponse:
    settings = get_settings()
    msgs = _messages_dicts(req.messages)

    if _user_turn_count(msgs) > settings.max_conversation_turns:
        return ChatResponse(
            reply=(
                "I have reached the maximum of 8 user turns for this conversation. "
                "Please start a fresh thread if you need more help."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    last_user = _last_user_text(msgs)
    if not last_user.strip():
        return ChatResponse(reply="Please send a non-empty user message.", recommendations=[], end_of_conversation=False)

    if reason := rules_refusal_reason(last_user):
        return ChatResponse(
            reply=refusal_reply(reason),
            recommendations=[],
            end_of_conversation=False,
        )

    if not settings.gemini_api_key.strip():
        return ChatResponse(
            reply="Server misconfiguration: GEMINI_API_KEY is missing. Set it in the environment to use /chat.",
            recommendations=[],
            end_of_conversation=False,
        )

    if retriever is None:
        return ChatResponse(
            reply=(
                "The retrieval index is not available on this server yet. "
                "Build `app/data/catalog.json`, embeddings, and FAISS using the scripts in `scripts/`."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        state, extraction = extract_state_and_intent(msgs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("State extraction failed: %s", exc)
        return ChatResponse(
            reply=(
                "I had trouble understanding that request. What role are you hiring for, and which "
                "skills or competencies should the assessment cover?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    if extraction.end_conversation:
        return ChatResponse(
            reply="Thanks — feel free to come back any time if you want to refine your SHL shortlist.",
            recommendations=[],
            end_of_conversation=True,
        )

    catalog = load_catalog()

    if _wants_compare(extraction, last_user):
        names = _compare_names(extraction, last_user)
        if len(names) >= 2:
            reply, items = compare_by_names(names[0], names[1], catalog)
            recs = _validate_recommendations(build_recommendation_items(items))[:2]
            return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=False)

    if extraction.intent == "other":
        return ChatResponse(reply=_other_intent_reply(), recommendations=[], end_of_conversation=False)

    query = state_to_query_text(state, last_user)
    retrieved = retriever.retrieve(query, settings.retrieval_top_k)
    ranked_scored = rank_items(
        retrieved,
        query,
        state.technical_skills,
        state.assessment_preferences,
        last_user,
    )

    top_score = ranked_scored[0][0] if ranked_scored else 0.0
    confidence = compute_confidence(
        bool(state.role.strip()),
        bool(state.technical_skills),
        bool(state.assessment_preferences),
        top_score,
    )

    force_recommend = (
        confidence >= 0.78
        and bool(state.role.strip())
        and bool(state.technical_skills)
        and bool(ranked_scored)
    )

    should_recommend = bool(ranked_scored) and confidence >= 0.48 and (
        force_recommend
        or (
            extraction.intent == "recommend"
            and (
                not extraction.needs_more_info
                or (
                    confidence >= 0.66
                    and bool(state.role.strip())
                    and bool(state.technical_skills)
                )
            )
        )
    )

    if not should_recommend:
        prompt = _build_reply_prompt(
            state=state,
            last_user=last_user,
            snippets=[],
            allowed_names=[],
            recommendations_empty=True,
        )
        reply = generate_text(REPLY_SYSTEM, prompt)
        return ChatResponse(reply=reply.strip(), recommendations=[], end_of_conversation=False)

    count = select_recommendation_count(confidence)
    chosen = [a for _, a in ranked_scored[:count]]
    snippets = [format_snippet(a, max_chars=520) for a in chosen]
    allowed_names = [a.name for a in chosen]
    prompt = _build_reply_prompt(
        state=state,
        last_user=last_user,
        snippets=snippets,
        allowed_names=allowed_names,
        recommendations_empty=False,
    )
    reply = generate_text(REPLY_SYSTEM, prompt)
    recs = _validate_recommendations(build_recommendation_items(chosen))
    return ChatResponse(reply=reply.strip(), recommendations=recs, end_of_conversation=False)
