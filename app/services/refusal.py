"""Lightweight refusal and injection guardrails (rules-first)."""

from __future__ import annotations

import re

REFUSAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bignore (all|previous) instructions\b", re.I), "policy"),
    (re.compile(r"\bsystem prompt\b", re.I), "policy"),
    (re.compile(r"\bjailbreak\b", re.I), "policy"),
    (re.compile(r"\byou are now\b", re.I), "policy"),
    (re.compile(r"\blegal advice\b", re.I), "scope"),
    (re.compile(r"\blawsuit\b", re.I), "scope"),
    (re.compile(r"\bvisa\b|\bimmigration\b", re.I), "scope"),
    (re.compile(r"\bmercer\b|\bhogan\b|\bpi\b test vendors\b", re.I), "scope"),
]


def rules_refusal_reason(message: str) -> str | None:
    for pat, kind in REFUSAL_PATTERNS:
        if pat.search(message):
            return kind
    return None


def refusal_reply(kind: str) -> str:
    if kind == "policy":
        return (
            "I can only help with SHL Individual Test Solutions from the official catalog. "
            "Tell me the role, skills, and whether you want ability, personality, or technical knowledge tests."
        )
    return (
        "I cannot help with that request. I only answer questions about SHL assessments listed in the "
        "Individual Test Solutions catalog (selection, comparison, and fit guidance)."
    )
