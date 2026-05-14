"""Google Gemini client helpers (text + JSON)."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import google.generativeai as genai
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def configure_genai() -> None:
    settings = get_settings()
    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=settings.gemini_api_key)


def generate_text(system_instruction: str, user_prompt: str) -> str:
    settings = get_settings()
    configure_genai()
    model = genai.GenerativeModel(
        settings.gemini_model,
        system_instruction=system_instruction,
    )
    resp = model.generate_content(
        user_prompt,
        request_options={"timeout": settings.request_timeout_seconds},
    )
    text = (resp.text or "").strip()
    return text


def generate_json_model(system_instruction: str, user_prompt: str, model_cls: type[T]) -> T:
    """Ask Gemini for strict JSON matching model_cls fields."""
    settings = get_settings()
    configure_genai()
    schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    prompt = (
        user_prompt
        + "\n\nReturn ONLY valid JSON with keys exactly matching this JSON Schema:\n"
        + schema_hint
    )
    model = genai.GenerativeModel(
        settings.gemini_model,
        system_instruction=system_instruction,
    )
    resp = model.generate_content(
        prompt,
        request_options={"timeout": settings.request_timeout_seconds},
    )
    raw = (resp.text or "").strip()
    raw = _strip_code_fence(raw)
    try:
        data: Any = json.loads(raw)
        return model_cls.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("JSON parse/validate failed (%s). Raw: %s", exc, raw[:500])
        raise


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
