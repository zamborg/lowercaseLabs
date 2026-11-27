"""Standalone HMAC-SHA256 caller identity signing and verification for ZOS modules.

This module uses only the Python standard library and has zero dependency on `zos`.
Modules may import `zos_module_sdk.caller` or copy this file verbatim into their own repo.

Two transports are supported using the same per-module `ZOS_MODULE_SECRET`:
1. MCP `params._meta["zos/caller"]` for `tools/call` requests (verified against `tool`
   and `args_digest` within ±60s skew). Calls without `zos/caller` are **system** calls
   (e.g. `tools/list`) and return `None`; modules must refuse to attribute them to a client.
2. HTTP `X-ZOS-Caller` + `X-ZOS-Caller-Signature` headers for `/admin/m/<slug>` UI proxy
   requests (`client_name = "owner"`, verified within ±60s skew).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MCP_CALLER_META_KEY = "zos/caller"
HTTP_CALLER_HEADER = "X-ZOS-Caller"
HTTP_CALLER_SIGNATURE_HEADER = "X-ZOS-Caller-Signature"
DEFAULT_MAX_SKEW_SECONDS = 60


class CallerVerificationError(ValueError):
    """Raised when a signed ZOS caller payload is malformed, tampered, or expired."""


@dataclass(frozen=True, slots=True)
class VerifiedCaller:
    """Verified caller identity extracted from a signed ZOS request."""

    principal_id: str
    client_id: str
    client_name: str
    ts: int


def canonical_json(data: Any) -> str:
    """Serialize JSON-compatible values deterministically."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_args_digest(args: Mapping[str, Any] | None) -> str:
    """Return lowercase hex SHA-256 of canonical JSON tool arguments."""
    payload = {} if args is None else dict(args)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _hmac_hex(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _check_skew(ts: int, *, max_skew_s: int, now: float | None) -> None:
    current = int(time.time() if now is None else now)
    if abs(current - ts) > max_skew_s:
        raise CallerVerificationError(
            f"Caller timestamp {ts} outside allowed ±{max_skew_s}s skew (now={current})"
        )


def sign_mcp_caller(
    secret: str,
    *,
    principal_id: str,
    client_id: str,
    client_name: str,
    tool: str,
    args: Mapping[str, Any] | None,
    ts: int | None = None,
) -> dict[str, Any]:
    """Build the signed `_meta['zos/caller']` envelope for an MCP `tools/call` request."""
    if not secret:
        raise ValueError("Module caller secret must be non-empty")
    issued_at = int(time.time() if ts is None else ts)
    payload = {
        "principal_id": str(principal_id),
        "client_id": str(client_id),
        "client_name": str(client_name),
        "tool": str(tool),
        "args_digest": compute_args_digest(args),
        "ts": issued_at,
    }
    sig = _hmac_hex(secret, canonical_json(payload).encode("utf-8"))
    return {"payload": payload, "sig": sig}


def verify_mcp_caller(
    meta: Mapping[str, Any] | None,
    *,
    secret: str,
    tool: str,
    args: Mapping[str, Any] | None,
    max_skew_s: int = DEFAULT_MAX_SKEW_SECONDS,
    now: float | None = None,
) -> VerifiedCaller | None:
    """Verify `_meta['zos/caller']` on an incoming MCP `tools/call`.

    Returns:
        - `None` if `_meta` is absent or contains no `zos/caller` key (unsigned system call).
        - `VerifiedCaller` if present and cryptographically valid for `(tool, args)` within skew.

    Raises:
        `CallerVerificationError` if `zos/caller` is present but invalid, tampered, mismatched,
        or outside the ±`max_skew_s` window.
    """
    if not meta or MCP_CALLER_META_KEY not in meta or meta[MCP_CALLER_META_KEY] is None:
        return None
    if not secret:
        raise CallerVerificationError("Module secret is not configured")

    envelope = meta[MCP_CALLER_META_KEY]
    if not isinstance(envelope, Mapping):
        raise CallerVerificationError("Malformed zos/caller envelope")
    payload = envelope.get("payload")
    sig = envelope.get("sig")
    if not isinstance(payload, Mapping) or not isinstance(sig, str) or not sig:
        raise CallerVerificationError("Missing payload or signature in zos/caller")

    expected_sig = _hmac_hex(secret, canonical_json(dict(payload)).encode("utf-8"))
    if not hmac.compare_digest(sig, expected_sig):
        raise CallerVerificationError("Invalid zos/caller HMAC signature")

    raw_ts = payload.get("ts")
    if isinstance(raw_ts, bool) or not isinstance(raw_ts, int):
        raise CallerVerificationError("Invalid caller timestamp")
    _check_skew(raw_ts, max_skew_s=max_skew_s, now=now)

    if str(payload.get("tool") or "") != tool:
        raise CallerVerificationError("Caller payload tool does not match request tool")

    expected_digest = compute_args_digest(args)
    if not hmac.compare_digest(str(payload.get("args_digest") or ""), expected_digest):
        raise CallerVerificationError("Caller payload args_digest does not match request arguments")

    principal_id = str(payload.get("principal_id") or "").strip()
    client_id = str(payload.get("client_id") or "").strip()
    client_name = str(payload.get("client_name") or "").strip()
    if not principal_id or not client_id or not client_name:
        raise CallerVerificationError("Incomplete caller identity fields")

    return VerifiedCaller(
        principal_id=principal_id,
        client_id=client_id,
        client_name=client_name,
        ts=raw_ts,
    )


def sign_http_caller(
    secret: str,
    *,
    principal_id: str,
    client_id: str = "owner",
    client_name: str = "owner",
    ts: int | None = None,
) -> tuple[str, str]:
    """Return `(X-ZOS-Caller, X-ZOS-Caller-Signature)` headers for a proxied module UI request."""
    if not secret:
        raise ValueError("Module caller secret must be non-empty")
    issued_at = int(time.time() if ts is None else ts)
    payload = {
        "principal_id": str(principal_id),
        "client_id": str(client_id),
        "client_name": str(client_name),
        "ts": issued_at,
    }
    raw_bytes = canonical_json(payload).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw_bytes).decode("ascii")
    sig = f"sha256={_hmac_hex(secret, encoded.encode('ascii'))}"
    return encoded, sig


def verify_http_caller(
    headers: Mapping[str, str],
    *,
    secret: str,
    max_skew_s: int = DEFAULT_MAX_SKEW_SECONDS,
    now: float | None = None,
) -> VerifiedCaller:
    """Verify `X-ZOS-Caller` and `X-ZOS-Caller-Signature` HTTP headers."""
    if not secret:
        raise CallerVerificationError("Module secret is not configured")
    lower = {k.lower(): v for k, v in headers.items()}
    raw_caller = lower.get(HTTP_CALLER_HEADER.lower(), "").strip()
    raw_sig = lower.get(HTTP_CALLER_SIGNATURE_HEADER.lower(), "").strip()
    if not raw_caller or not raw_sig:
        raise CallerVerificationError("Missing X-ZOS-Caller or X-ZOS-Caller-Signature header")

    sig_hex = raw_sig[len("sha256=") :] if raw_sig.startswith("sha256=") else raw_sig

    expected_sig = _hmac_hex(secret, raw_caller.encode("ascii"))
    if not hmac.compare_digest(sig_hex, expected_sig):
        raise CallerVerificationError("Invalid X-ZOS-Caller-Signature")

    try:
        decoded = base64.urlsafe_b64decode(raw_caller.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise CallerVerificationError("Malformed X-ZOS-Caller header payload") from exc

    if not isinstance(payload, dict):
        raise CallerVerificationError("X-ZOS-Caller payload must be a JSON object")

    raw_ts = payload.get("ts")
    if isinstance(raw_ts, bool) or not isinstance(raw_ts, int):
        raise CallerVerificationError("Invalid caller timestamp")
    _check_skew(raw_ts, max_skew_s=max_skew_s, now=now)

    principal_id = str(payload.get("principal_id") or "").strip()
    client_id = str(payload.get("client_id") or "").strip()
    client_name = str(payload.get("client_name") or "").strip()
    if not principal_id or not client_id or not client_name:
        raise CallerVerificationError("Incomplete HTTP caller identity fields")

    return VerifiedCaller(
        principal_id=principal_id,
        client_id=client_id,
        client_name=client_name,
        ts=raw_ts,
    )
