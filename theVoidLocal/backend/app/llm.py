from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from .config import settings
from .tagging import TAG_CATALOG, clean_tags, contains_phrase, extract_mood_tags, extract_themes, normalize_tag

POSITIVE_WORDS = {
    "good",
    "great",
    "happy",
    "better",
    "calm",
    "grateful",
    "excited",
    "hopeful",
}
NEGATIVE_WORDS = {
    "bad",
    "sad",
    "stressed",
    "anxious",
    "angry",
    "tired",
    "overwhelmed",
    "lonely",
}

SAFETY_TERMS = {
    "suicide",
    "kill myself",
    "self harm",
    "hopeless",
    "want to disappear",
}

logger = logging.getLogger(__name__)
_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI | None:
    global _openai_client
    if not settings.openai_api_key:
        return None
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_request_timeout_seconds,
        )
    return _openai_client


def _tags_from_payload(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("tags")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str)]
    return []


def _extract_tags_with_llm(transcript_text: str) -> list[str] | None:
    client = _get_openai_client()
    if client is None:
        return None

    try:
        completion = client.chat.completions.create(
            model=settings.openai_insights_model,
            temperature=0.2,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You tag short voice reflection transcripts.\n"
                        "Select between 1 and 4 tags.\n"
                        "Use only tags from the allowed vocabulary.\n"
                        "Return strict JSON: {\"tags\": [\"tag1\", \"tag2\"]}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Allowed tags: {', '.join(TAG_CATALOG)}\n\n"
                        f"Transcript:\n{transcript_text}"
                    ),
                },
            ],
        )
    except Exception:
        logger.exception("openai_tagging_failed")
        return None

    content = completion.choices[0].message.content if completion.choices else None
    if not isinstance(content, str) or not content.strip():
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("openai_tagging_non_json_response")
        return None

    if not isinstance(parsed, dict):
        return None

    raw_tags = _tags_from_payload(parsed)
    if not raw_tags:
        return None

    if all(normalize_tag(tag) is None for tag in raw_tags):
        return None

    cleaned = clean_tags(raw_tags, minimum=1, maximum=4)
    return cleaned or None


def _score_tokens(lowered: str) -> tuple[int, int]:
    pos = sum(1 for word in POSITIVE_WORDS if contains_phrase(lowered, word))
    neg = sum(1 for word in NEGATIVE_WORDS if contains_phrase(lowered, word))
    return pos, neg


def _mood_score(lowered: str) -> float:
    pos, neg = _score_tokens(lowered)
    raw = pos - neg
    if raw <= -4:
        return -2.0
    if raw == -3:
        return -1.5
    if raw == -2:
        return -1.0
    if raw == -1:
        return -0.5
    if raw == 0:
        return 0.0
    if raw == 1:
        return 0.5
    if raw == 2:
        return 1.0
    if raw == 3:
        return 1.5
    return 2.0


def _summary(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return "Short reflection captured."
    first = " ".join(tokens[:24])
    if len(tokens) <= 24:
        return first
    second = " ".join(tokens[24:42])
    return f"{first}... {second}..."


def _signals(lowered: str) -> dict:
    stress = (
        0.7
        if any(contains_phrase(lowered, word) for word in {"stress", "stressed", "anxious", "overwhelmed"})
        else 0.3
    )
    energy = (
        0.7
        if any(contains_phrase(lowered, word) for word in {"excited", "energetic", "motivated"})
        else 0.4
    )
    confidence = (
        0.7
        if any(contains_phrase(lowered, word) for word in {"confident", "sure", "ready"})
        else 0.4
    )
    return {
        "stress": stress,
        "energy": energy,
        "confidence": confidence,
    }


def _safety_flags(lowered: str) -> dict:
    mention = any(contains_phrase(lowered, term) for term in SAFETY_TERMS)
    return {
        "needs_review": mention,
        "matched_terms": [term for term in SAFETY_TERMS if contains_phrase(lowered, term)],
    }


def extract_insights(transcript_text: str) -> dict:
    lowered = transcript_text.lower()
    llm_tags = _extract_tags_with_llm(transcript_text)
    fallback_tags = extract_mood_tags(lowered, limit=4)
    return {
        "mood_score": _mood_score(lowered),
        "mood_tags": llm_tags or fallback_tags,
        "summary": _summary(transcript_text),
        "themes": extract_themes(lowered),
        "signals": _signals(lowered),
        "safety_flags": _safety_flags(lowered),
    }
