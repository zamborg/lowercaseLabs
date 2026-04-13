from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.config import get_settings


class AudioStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.root = self.settings.absolute_storage_root / "audio"
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, *, user_id: str, filename: str | None, payload: bytes) -> str:
        safe_name = (filename or "capture.m4a").replace("/", "-")
        object_key = f"{user_id}/{uuid4()}-{safe_name}"
        path = self.path_for(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return object_key

    def read_bytes(self, object_key: str) -> bytes:
        return self.path_for(object_key).read_bytes()

    def delete(self, object_key: str | None) -> None:
        if not object_key:
            return
        path = self.path_for(object_key)
        if path.exists():
            path.unlink()

    def path_for(self, object_key: str) -> Path:
        return self.root / object_key

