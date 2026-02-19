from __future__ import annotations

from hashlib import sha256

import pytest
from fastapi import HTTPException

from app import auth
from app.config import settings


def _raise_identity_error(_: str, nonce: str | None = None) -> str:
    del nonce
    raise auth.AppleIdentityTokenError("invalid")


def test_verify_apple_identity_token_validates_signature_claims_and_nonce(monkeypatch) -> None:
    monkeypatch.setattr(settings, "apple_allowed_audiences", "lowercaseLabs.theVoid")
    monkeypatch.setattr(auth, "_resolve_apple_jwk", lambda kid: {"kid": kid})
    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda _: {"kid": "kid-1", "alg": "RS256"})

    captured: dict[str, object] = {}

    def fake_decode(
        token: str,
        key: dict,
        algorithms: list[str],
        audience: str,
        issuer: str,
        options: dict,
    ) -> dict:
        captured["token"] = token
        captured["key"] = key
        captured["algorithms"] = algorithms
        captured["audience"] = audience
        captured["issuer"] = issuer
        captured["options"] = options
        return {"sub": "apple-user-123", "nonce": sha256("raw-nonce".encode("utf-8")).hexdigest()}

    monkeypatch.setattr(auth.jwt, "decode", fake_decode)

    subject = auth.verify_apple_identity_token("token-value", nonce="raw-nonce")

    assert subject == "apple-user-123"
    assert captured["audience"] == "lowercaseLabs.theVoid"
    assert captured["issuer"] == settings.apple_issuer
    assert captured["algorithms"] == ["RS256"]


def test_verify_apple_identity_token_tries_multiple_allowed_audiences(monkeypatch) -> None:
    monkeypatch.setattr(settings, "apple_allowed_audiences", "wrong.aud,com.lowercaseLabs.theVoid")
    monkeypatch.setattr(auth, "_resolve_apple_jwk", lambda kid: {"kid": kid})
    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda _: {"kid": "kid-1", "alg": "RS256"})

    seen_audiences: list[str] = []

    def fake_decode(
        token: str,
        key: dict,
        algorithms: list[str],
        audience: str,
        issuer: str,
        options: dict,
    ) -> dict:
        del token, key, algorithms, issuer, options
        seen_audiences.append(audience)
        if audience == "wrong.aud":
            raise auth.JWTError("invalid audience")
        return {"sub": "apple-user-456"}

    monkeypatch.setattr(auth.jwt, "decode", fake_decode)

    subject = auth.verify_apple_identity_token("token-value", nonce=None)

    assert subject == "apple-user-456"
    assert seen_audiences == ["wrong.aud", "com.lowercaseLabs.theVoid"]


def test_parse_apple_subject_allows_dev_tokens_in_non_production(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "allow_dev_identity_tokens", True)
    monkeypatch.setattr(auth, "verify_apple_identity_token", _raise_identity_error)

    first = auth.parse_apple_subject("dev-user-a")
    second = auth.parse_apple_subject("dev-user-a")

    assert first == second
    assert first.startswith("dev-")


def test_parse_apple_subject_rejects_non_dev_unverified_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "allow_dev_identity_tokens", True)
    monkeypatch.setattr(auth, "verify_apple_identity_token", _raise_identity_error)

    with pytest.raises(HTTPException) as exc:
        auth.parse_apple_subject("not-a-dev-token")

    assert exc.value.status_code == 401


def test_parse_apple_subject_disables_dev_fallback_in_production(monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "allow_dev_identity_tokens", True)
    monkeypatch.setattr(auth, "verify_apple_identity_token", _raise_identity_error)

    with pytest.raises(HTTPException) as exc:
        auth.parse_apple_subject("dev-user-a")

    assert exc.value.status_code == 401
