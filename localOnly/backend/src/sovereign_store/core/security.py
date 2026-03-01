from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from sovereign_store.core.config import get_settings

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    user_id: UUID
    pod_access_list: list[UUID] = Field(default_factory=list)
    exp: int | None = None


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_context.verify(password, password_hash)


def create_access_token(
    *, user_id: UUID, pod_access_list: list[UUID], expires_delta: timedelta | None = None
) -> str:
    settings = get_settings()
    expiry = datetime.now(tz=UTC) + (
        expires_delta or timedelta(minutes=settings.jwt_exp_minutes)
    )
    payload = {
        "user_id": str(user_id),
        "pod_access_list": [str(pod_id) for pod_id in pod_access_list],
        "iat": int(datetime.now(tz=UTC).timestamp()),
        "exp": int(expiry.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_signing_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> TokenPayload:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_verification_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        raise ValueError("Invalid token") from exc
    return TokenPayload(**payload)
