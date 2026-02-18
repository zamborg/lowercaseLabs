from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
import threading
import time
import secrets
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, now_utc
from .models import User

bearer_scheme = HTTPBearer(auto_error=False)
_apple_jwks_lock = threading.Lock()
_apple_jwks_by_kid: dict[str, dict] = {}
_apple_jwks_expires_at = 0.0


class AppleIdentityTokenError(ValueError):
    pass


def _random_handle() -> str:
    return f"void-{secrets.token_hex(3)}"


def _fetch_apple_jwks() -> dict[str, dict]:
    try:
        response = httpx.get(
            settings.apple_jwks_url,
            timeout=settings.apple_jwks_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise AppleIdentityTokenError("Could not fetch Apple signing keys") from exc

    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, list):
        raise AppleIdentityTokenError("Apple signing keys are unavailable")

    keys_by_kid: dict[str, dict] = {}
    for raw_key in raw_keys:
        if not isinstance(raw_key, dict):
            continue
        kid = raw_key.get("kid")
        if isinstance(kid, str) and kid.strip():
            keys_by_kid[kid.strip()] = raw_key

    if not keys_by_kid:
        raise AppleIdentityTokenError("Apple signing keys are unavailable")
    return keys_by_kid


def _refresh_apple_jwks_cache(force: bool = False) -> None:
    global _apple_jwks_by_kid, _apple_jwks_expires_at

    now = time.time()
    with _apple_jwks_lock:
        if (
            not force
            and _apple_jwks_by_kid
            and now < _apple_jwks_expires_at
        ):
            return

        _apple_jwks_by_kid = _fetch_apple_jwks()
        cache_ttl = max(60, int(settings.apple_jwks_cache_ttl_seconds))
        _apple_jwks_expires_at = now + cache_ttl


def _resolve_apple_jwk(kid: str) -> dict:
    _refresh_apple_jwks_cache()

    with _apple_jwks_lock:
        cached_key = _apple_jwks_by_kid.get(kid)
    if cached_key is not None:
        return cached_key

    _refresh_apple_jwks_cache(force=True)
    with _apple_jwks_lock:
        refreshed_key = _apple_jwks_by_kid.get(kid)
    if refreshed_key is None:
        raise AppleIdentityTokenError("Unknown Apple signing key")
    return refreshed_key


def _validate_apple_nonce(claims: dict, nonce: str | None) -> None:
    if nonce is None:
        return

    raw_nonce = nonce.strip()
    if not raw_nonce:
        return

    claim_nonce = claims.get("nonce")
    if not isinstance(claim_nonce, str) or not claim_nonce.strip():
        raise AppleIdentityTokenError("Missing nonce claim")

    expected_hash = sha256(raw_nonce.encode("utf-8")).hexdigest()
    if claim_nonce not in {raw_nonce, expected_hash}:
        raise AppleIdentityTokenError("Nonce mismatch")


def verify_apple_identity_token(identity_token: str, nonce: str | None = None) -> str:
    if not settings.apple_audiences:
        raise AppleIdentityTokenError("APPLE_ALLOWED_AUDIENCES is not configured")

    try:
        headers = jwt.get_unverified_header(identity_token)
    except JWTError as exc:
        raise AppleIdentityTokenError("Malformed identity token") from exc

    kid = headers.get("kid")
    alg = headers.get("alg")
    if alg != "RS256":
        raise AppleIdentityTokenError("Unexpected token algorithm")
    if not isinstance(kid, str) or not kid.strip():
        raise AppleIdentityTokenError("Missing signing key id")

    key = _resolve_apple_jwk(kid.strip())
    audience: str | list[str]
    if len(settings.apple_audiences) == 1:
        audience = settings.apple_audiences[0]
    else:
        audience = list(settings.apple_audiences)

    try:
        claims = jwt.decode(
            identity_token,
            key,
            algorithms=["RS256"],
            audience=audience,
            issuer=settings.apple_issuer,
            options={"verify_aud": True},
        )
    except JWTError as exc:
        raise AppleIdentityTokenError("Token signature or claims are invalid") from exc

    _validate_apple_nonce(claims, nonce)

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        raise AppleIdentityTokenError("Missing subject claim")
    return sub.strip()


def _dev_subject(identity_token: str) -> str:
    return f"dev-{sha256(identity_token.encode('utf-8')).hexdigest()}"


def parse_apple_subject(identity_token: str, nonce: str | None = None) -> str:
    normalized_token = identity_token.strip()
    if not normalized_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing identity token")

    try:
        return verify_apple_identity_token(normalized_token, nonce=nonce)
    except AppleIdentityTokenError as exc:
        if settings.dev_identity_fallback_enabled and normalized_token.startswith("dev-"):
            return _dev_subject(normalized_token)
        detail = "Invalid Apple identity token"
        if not settings.is_production:
            detail = f"{detail}: {exc}"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail) from exc


def create_access_token(user_id: str) -> str:
    expires = now_utc() + timedelta(minutes=settings.access_token_exp_minutes)
    payload = {
        "sub": user_id,
        "exp": int(expires.timestamp()),
        "iat": int(now_utc().timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def issue_session_for_apple_token(
    db: Session,
    identity_token: str,
    nonce: str | None = None,
    display_name: str | None = None,
    daily_checkin_time_local: str | None = None,
    timezone: str | None = None,
) -> tuple[User, str]:
    apple_sub = parse_apple_subject(identity_token, nonce=nonce)
    user = db.query(User).filter(User.apple_sub == apple_sub).one_or_none()

    if user is None:
        handle = _random_handle()
        while db.query(User).filter(User.anonymous_handle == handle).first() is not None:
            handle = _random_handle()
        user = User(
            apple_sub=apple_sub,
            display_name=display_name,
            anonymous_handle=handle,
            daily_checkin_time_local=daily_checkin_time_local or "20:00",
            timezone=timezone or "UTC",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if display_name is not None:
            user.display_name = display_name
        if daily_checkin_time_local is not None:
            user.daily_checkin_time_local = daily_checkin_time_local
        if timezone is not None:
            user.timezone = timezone
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(user.id)
    return user, token


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user_id = payload.get("sub")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return user_id


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    user_id = decode_access_token(credentials.credentials)
    user = db.query(User).filter(User.id == user_id).one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
