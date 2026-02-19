from __future__ import annotations

from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

from fastapi import HTTPException, status
from jose import JWTError, jwt

from .config import settings
from .db import now_utc


class LocalObjectStorage:
    def __init__(self, root: str, signing_secret: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.signing_secret = signing_secret

    def _resolve_object_path(self, object_key: str) -> Path:
        key_path = PurePosixPath(object_key)
        if key_path.is_absolute() or ".." in key_path.parts:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid object key")

        path = self.root.joinpath(*key_path.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def create_object_key(self, user_id: str, entry_id: str, extension: str = "m4a") -> str:
        ext = extension.strip(".") or "m4a"
        return f"{user_id}/{entry_id}.{ext}"

    def create_upload_token(self, user_id: str, object_key: str, ttl_seconds: int | None = None) -> tuple[str, int]:
        ttl = ttl_seconds or settings.upload_token_ttl_seconds
        expires = now_utc() + timedelta(seconds=ttl)
        payload = {
            "sub": user_id,
            "object_key": object_key,
            "exp": int(expires.timestamp()),
            "iat": int(now_utc().timestamp()),
        }
        token = jwt.encode(payload, self.signing_secret, algorithm=settings.jwt_algorithm)
        return token, int(expires.timestamp())

    def verify_upload_token(self, token: str, user_id: str, object_key: str) -> None:
        try:
            payload = jwt.decode(token, self.signing_secret, algorithms=[settings.jwt_algorithm])
        except JWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid upload token") from exc

        if payload.get("sub") != user_id or payload.get("object_key") != object_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Upload token mismatch")

    def upload_url(self, base_url: str, object_key: str, token: str) -> str:
        return f"{base_url}?{urlencode({'token': token})}"

    def write_bytes(self, object_key: str, payload: bytes) -> None:
        object_path = self._resolve_object_path(object_key)
        object_path.write_bytes(payload)

    def object_exists(self, object_key: str) -> bool:
        object_path = self._resolve_object_path(object_key)
        return object_path.exists()

    def read_bytes(self, object_key: str) -> bytes:
        object_path = self._resolve_object_path(object_key)
        if not object_path.exists():
            raise FileNotFoundError(object_key)
        return object_path.read_bytes()

    def delete_object(self, object_key: str) -> bool:
        object_path = self._resolve_object_path(object_key)
        if not object_path.exists():
            return False
        object_path.unlink()
        return True
