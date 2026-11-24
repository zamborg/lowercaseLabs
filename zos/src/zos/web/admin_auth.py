"""Signed HttpOnly cookie sessions and CSRF verification for /admin."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import Response

ADMIN_SESSION_COOKIE = "zos_admin_session"
ADMIN_SESSION_TTL_SECONDS = 12 * 3600  # 12 hours

ADMIN_SECURITY_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def derive_session_signing_key(*, secret_key: str, owner_password_hash: str) -> bytes:
    """Derive a dedicated 32-byte HMAC-SHA256 signing key for /admin session cookies."""
    seed = secret_key or owner_password_hash or secrets.token_hex(32)
    return hashlib.sha256(b"zos-admin-session-v1:" + seed.encode("utf-8")).digest()


@dataclass(frozen=True, slots=True)
class AdminSession:
    issued_at: int
    expires_at: int
    csrf_token: str


class AdminSessionManager:
    """Signs and verifies stateless HttpOnly HMAC-SHA256 admin cookies."""

    def __init__(
        self,
        *,
        signing_key: bytes,
        secure_cookie: bool,
        ttl_seconds: int = ADMIN_SESSION_TTL_SECONDS,
    ) -> None:
        self._key = signing_key
        self.secure_cookie = secure_cookie
        self.ttl_seconds = ttl_seconds

    def create_session(self, *, now: float | None = None) -> tuple[AdminSession, str]:
        ts = int(time.time() if now is None else now)
        session = AdminSession(
            issued_at=ts,
            expires_at=ts + self.ttl_seconds,
            csrf_token=secrets.token_urlsafe(32),
        )
        token = self.sign(session)
        return session, token

    def sign(self, session: AdminSession) -> str:
        payload_bytes = json.dumps(
            {
                "iat": session.issued_at,
                "exp": session.expires_at,
                "csrf": session.csrf_token,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
        sig = hmac.new(self._key, payload_b64.encode("ascii"), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
        return f"{payload_b64}.{sig_b64}"

    def verify(self, cookie_value: str | None, *, now: float | None = None) -> AdminSession | None:
        if not cookie_value or "." not in cookie_value:
            return None
        parts = cookie_value.split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts
        expected_sig = hmac.new(self._key, payload_b64.encode("ascii"), hashlib.sha256).digest()
        expected_b64 = base64.urlsafe_b64encode(expected_sig).decode("ascii").rstrip("=")
        if not hmac.compare_digest(sig_b64, expected_b64):
            return None

        try:
            padded = payload_b64 + "=" * ((-len(payload_b64)) % 4)
            data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
            exp = int(data["exp"])
            iat = int(data["iat"])
            csrf = str(data["csrf"])
        except Exception:
            return None

        current = time.time() if now is None else now
        if not csrf or exp <= current:
            return None
        return AdminSession(issued_at=iat, expires_at=exp, csrf_token=csrf)

    def read_session(self, request: Request, *, now: float | None = None) -> AdminSession | None:
        return self.verify(request.cookies.get(ADMIN_SESSION_COOKIE), now=now)

    def set_cookie(self, response: Response, cookie_token: str) -> None:
        response.set_cookie(
            key=ADMIN_SESSION_COOKIE,
            value=cookie_token,
            max_age=self.ttl_seconds,
            path="/admin",
            httponly=True,
            samesite="strict",
            secure=self.secure_cookie,
        )

    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(
            key=ADMIN_SESSION_COOKIE,
            path="/admin",
            httponly=True,
            samesite="strict",
            secure=self.secure_cookie,
        )

    @staticmethod
    def verify_csrf(session: AdminSession, submitted_token: str | None) -> bool:
        if not submitted_token:
            return False
        return hmac.compare_digest(session.csrf_token, submitted_token)
