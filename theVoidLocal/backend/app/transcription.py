from __future__ import annotations

import io
import logging
from pathlib import PurePosixPath
from typing import Any

from openai import OpenAI

from .config import settings
from .models import Entry
from .storage import LocalObjectStorage

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _file_name_for_object_key(object_key: str, fallback_entry_id: str) -> str:
    name = PurePosixPath(object_key).name
    if not name:
        return f"{fallback_entry_id}.m4a"
    return name


def _extract_transcript_text(response: Any) -> str:
    if isinstance(response, str):
        text = response.strip()
        if text:
            return text
        raise RuntimeError("OpenAI transcription returned empty text")

    text = getattr(response, "text", None)
    if isinstance(text, str):
        text = text.strip()
        if text:
            return text

    if isinstance(response, dict):
        dict_text = response.get("text")
        if isinstance(dict_text, str):
            dict_text = dict_text.strip()
            if dict_text:
                return dict_text

    raise RuntimeError("OpenAI transcription response had no text field")


def _openai_metadata(response: Any, byte_count: int) -> dict:
    metadata: dict[str, Any] = {
        "provider": "openai",
        "model": settings.openai_transcription_model,
        "byte_count": byte_count,
    }

    language = getattr(response, "language", None)
    if isinstance(language, str) and language:
        metadata["detected_language"] = language

    duration = getattr(response, "duration", None)
    if isinstance(duration, (float, int)):
        metadata["detected_duration_seconds"] = float(duration)

    segments = getattr(response, "segments", None)
    if isinstance(segments, list):
        metadata["segment_count"] = len(segments)

    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            metadata["usage"] = usage.model_dump(exclude_none=True)
        elif isinstance(usage, dict):
            metadata["usage"] = usage

    return metadata


def _stub_transcript(entry: Entry, audio_bytes: bytes) -> tuple[str, dict]:
    byte_count = len(audio_bytes)
    transcript = (
        "Voice note captured in theVoid. "
        f"Duration: {entry.duration_seconds}s. "
        f"Approx bytes: {byte_count}. "
        "(Set OPENAI_API_KEY to enable real transcription.)"
    )
    metadata = {
        "provider": "stub",
        "byte_count": byte_count,
        "duration_seconds": entry.duration_seconds,
    }
    return transcript, metadata


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


def _transcribe_with_openai(entry: Entry, audio_bytes: bytes) -> tuple[str, dict]:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable")

    audio_stream = io.BytesIO(audio_bytes)
    audio_stream.name = _file_name_for_object_key(entry.audio_object_key, entry.id)

    request: dict[str, Any] = {
        "model": settings.openai_transcription_model,
        "file": audio_stream,
    }
    if settings.openai_transcription_language:
        request["language"] = settings.openai_transcription_language

    response = client.audio.transcriptions.create(**request)
    text = _extract_transcript_text(response)
    metadata = _openai_metadata(response, len(audio_bytes))
    return text, metadata


def transcribe_entry(entry: Entry, storage: LocalObjectStorage) -> tuple[str, dict]:
    audio_bytes = storage.read_bytes(entry.audio_object_key)
    if not audio_bytes:
        raise RuntimeError(f"Audio payload empty for entry {entry.id}")

    if settings.openai_api_key:
        try:
            return _transcribe_with_openai(entry, audio_bytes)
        except Exception:
            logger.exception("openai_transcription_failed", extra={"entry_id": entry.id})
            raise

    return _stub_transcript(entry, audio_bytes)
