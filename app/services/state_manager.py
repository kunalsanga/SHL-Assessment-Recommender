"""Structured conversation state extraction (full snapshot each request)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.prompts.templates import STATE_EXTRACTION_SYSTEM, state_user_block
from app.services.gemini_client import generate_json_model


class ConversationState(BaseModel):
    role: str = ""
    seniority: str = ""
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    assessment_preferences: list[str] = Field(default_factory=list)


class StateExtraction(BaseModel):
    role: str = ""
    seniority: str = ""
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    assessment_preferences: list[str] = Field(default_factory=list)
    intent: Literal["clarify", "recommend", "compare", "other"] = "clarify"
    compare_names: list[str] = Field(default_factory=list)
    needs_more_info: bool = True
    end_conversation: bool = False


def _dedupe_list(values: list[str], max_items: int = 24) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in values:
        t = x.strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def snapshot_from_extraction(extracted: StateExtraction) -> ConversationState:
    return ConversationState(
        role=extracted.role.strip(),
        seniority=extracted.seniority.strip(),
        technical_skills=_dedupe_list(extracted.technical_skills),
        soft_skills=_dedupe_list(extracted.soft_skills),
        assessment_preferences=_dedupe_list(extracted.assessment_preferences),
    )


def extract_state_and_intent(messages: list[dict[str, str]]) -> tuple[ConversationState, StateExtraction]:
    block = state_user_block(messages)
    user_prompt = (
        "Infer the best structured state implied by the ENTIRE conversation (latest message wins "
        "if there is a correction).\n\n"
        f"{block}\n"
    )
    extracted = generate_json_model(STATE_EXTRACTION_SYSTEM, user_prompt, StateExtraction)
    return snapshot_from_extraction(extracted), extracted


def state_to_query_text(state: ConversationState, last_user: str) -> str:
    parts = [
        last_user.strip(),
        state.role,
        state.seniority,
        " ".join(state.technical_skills),
        " ".join(state.soft_skills),
        " ".join(state.assessment_preferences),
    ]
    return " ".join(p for p in parts if p).strip()
