from __future__ import annotations

from datetime import date

from app.config import settings
from app.models import Entry
from app.storage import LocalObjectStorage
from app.transcription import transcribe_entry


def test_transcribe_entry_falls_back_to_stub_without_openai_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", None)

    storage = LocalObjectStorage(str(tmp_path), "test-secret")
    object_key = "user-1/entry-1.m4a"
    storage.write_bytes(object_key, b"sample-audio-bytes")

    entry = Entry(
        id="entry-1",
        user_id="user-1",
        local_date=date.today(),
        audio_object_key=object_key,
        duration_seconds=42,
        trace_id="trace-1",
    )

    transcript, metadata = transcribe_entry(entry, storage)

    assert "Set OPENAI_API_KEY" in transcript
    assert metadata["provider"] == "stub"
    assert metadata["duration_seconds"] == 42
