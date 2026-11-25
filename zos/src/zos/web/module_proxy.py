"""Reverse proxy mounting installed module human UIs at `/admin/m/<slug>/{path}` (M10c).

Security model:
- Gated behind the owner `/admin` session cookie (`zos_admin_session`).
- Strips `zos_admin_session` (and any `zos_admin*` cookie) and `Authorization` before forwarding.
- Injects HMAC-SHA256-signed `X-ZOS-Caller` + `X-ZOS-Caller-Signature` (`client_name="owner"`)
  using the module's per-module secret, plus `X-Forwarded-Prefix: /admin/m/<slug>`.
- Rewrites `Location` headers under `/admin/m/<slug>` and scopes module `Set-Cookie` headers
  to `Path=/admin/m/<slug>`, dropping any cookie named like the admin session.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import AsyncIterator, Callable
from http.cookies import SimpleCookie
from urllib.parse import urlsplit, urlunsplit

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response, StreamingResponse
from zos_module_sdk.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    sign_http_caller,
)

from zos.modules import ModuleManager
from zos.web.admin_auth import ADMIN_SESSION_COOKIE

logger = logging.getLogger(__name__)

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

_STRIPPED_REQUEST_HEADERS = _HOP_BY_HOP_HEADERS | frozenset(
    {
        "authorization",
        "cookie",
        "accept-encoding",
        "x-forwarded-prefix",
        "x-forwarded-proto",
        "x-forwarded-for",
        "x-forwarded-host",
        HTTP_CALLER_HEADER.lower(),
        HTTP_CALLER_SIGNATURE_HEADER.lower(),
    }
)

_STRIPPED_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | frozenset(
    {
        "set-cookie",
        "location",
        "content-encoding",
    }
)

_COOKIE_PATH_RE = re.compile(r"(;\s*Path=)([^;]*)", flags=re.IGNORECASE)


def strip_admin_cookies(raw_cookie_header: str | None) -> str | None:
    """Remove `zos_admin_session` (and any `zos_admin*` cookie) from an inbound `Cookie` header."""
    if not raw_cookie_header:
        return None
    kept: list[str] = []
    for part in raw_cookie_header.split(";"):
        piece = part.strip()
        if not piece or "=" not in piece:
            continue
        name = piece.split("=", 1)[0].strip().lower()
        if name == ADMIN_SESSION_COOKIE.lower() or name.startswith("zos_admin"):
            continue
        kept.append(piece)
    return "; ".join(kept) if kept else None


def rewrite_location_header(location: str, *, prefix: str, upstream_base_url: str) -> str:
    """Rewrite upstream `Location` redirects so they stay under `/admin/m/<slug>`."""
    clean_prefix = prefix.rstrip("/")
    raw = location.strip()
    if not raw:
        return raw

    # Already prefixed
    if raw == clean_prefix or raw.startswith(clean_prefix + "/"):
        return raw

    # Root-relative path (`/foo?a=1#h`)
    if raw.startswith("/") and not raw.startswith("//"):
        return f"{clean_prefix}{raw}"

    # Absolute URL targeting the module's own upstream origin
    parsed_loc = urlsplit(raw)
    parsed_up = urlsplit(upstream_base_url)
    if (
        parsed_loc.scheme in ("http", "https")
        and parsed_loc.netloc
        and parsed_loc.netloc.lower() == parsed_up.netloc.lower()
    ):
        path = parsed_loc.path or "/"
        if not path.startswith("/"):
            path = "/" + path
        if not (path == clean_prefix or path.startswith(clean_prefix + "/")):
            path = f"{clean_prefix}{path}"
        return urlunsplit(("", "", path, parsed_loc.query, parsed_loc.fragment))

    return raw


def filter_and_scope_set_cookie(
    raw_set_cookie: str,
    *,
    prefix: str,
    secure_cookie: bool = False,
) -> str | None:
    """Drop admin-named cookies and enforce `Path=/admin/m/<slug>` on allowed module cookies."""
    clean_prefix = prefix.rstrip("/")
    first_pair = raw_set_cookie.split(";", 1)[0].strip()
    if "=" not in first_pair:
        return None
    cookie_name = first_pair.split("=", 1)[0].strip().lower()
    if not cookie_name or cookie_name == ADMIN_SESSION_COOKIE.lower() or cookie_name.startswith("zos_admin"):
        return None

    # Validate syntax lightly via SimpleCookie when possible
    parsed = SimpleCookie()
    with contextlib.suppress(Exception):
        parsed.load(raw_set_cookie)

    m = _COOKIE_PATH_RE.search(raw_set_cookie)
    if m is not None:
        existing_path = m.group(2).strip()
        if existing_path == clean_prefix or existing_path.startswith(clean_prefix + "/"):
            scoped = raw_set_cookie
        else:
            scoped = _COOKIE_PATH_RE.sub(rf"\g<1>{clean_prefix}", raw_set_cookie, count=1)
    else:
        scoped = f"{raw_set_cookie}; Path={clean_prefix}"

    if secure_cookie and "; secure" not in scoped.lower():
        scoped = f"{scoped}; Secure"
    return scoped


async def proxy_module_ui_request(
    request: Request,
    *,
    slug: str,
    subpath: str,
    module_manager: ModuleManager,
    base_url: str,
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> Response:
    """Stream-proxy an authenticated owner request to an installed module's `ui_url`."""
    clean_slug = slug.strip().lower()
    mod = await module_manager.get_module(clean_slug)
    if (
        mod is None
        or mod.get("disabled_at") is not None
        or mod.get("status") == "disabled"
        or not mod.get("ui_url")
    ):
        return HTMLResponse("<h1>Module UI not found</h1>", status_code=404)

    ui_base = str(mod["ui_url"]).rstrip("/")
    prefix = f"/admin/m/{clean_slug}"
    rel_path = subpath.lstrip("/")
    target_url = f"{ui_base}/{rel_path}" if rel_path else f"{ui_base}/"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    secret = await module_manager.get_module_secret(
        clean_slug,
        principal_id=mod["principal_id"],
    )
    if not secret:
        return HTMLResponse("<h1>Module caller secret unavailable</h1>", status_code=502)

    caller_hdr, sig_hdr = sign_http_caller(
        secret,
        principal_id=str(mod["principal_id"]),
        client_id="owner",
        client_name="owner",
    )

    fwd_headers: dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() in _STRIPPED_REQUEST_HEADERS:
            continue
        fwd_headers[k] = v

    filtered_cookie = strip_admin_cookies(request.headers.get("cookie"))
    if filtered_cookie:
        fwd_headers["Cookie"] = filtered_cookie

    fwd_headers["Accept-Encoding"] = "identity"
    fwd_headers["X-Forwarded-Prefix"] = prefix
    fwd_headers["X-Forwarded-Proto"] = "https" if base_url.lower().startswith("https://") else "http"
    fwd_headers[HTTP_CALLER_HEADER] = caller_hdr
    fwd_headers[HTTP_CALLER_SIGNATURE_HEADER] = sig_hdr

    body = await request.body()
    client = (
        http_client_factory()
        if http_client_factory is not None
        else httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, read=60.0),
        )
    )

    try:
        upstream_req = client.build_request(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=body if body else None,
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except Exception as exc:
        await client.aclose()
        logger.warning("Module UI proxy failed for '%s': %s", clean_slug, exc)
        return HTMLResponse(
            f"<h1>Module '{clean_slug}' UI unreachable</h1><p>Could not connect to module UI.</p>",
            status_code=502,
        )

    secure_cookie = base_url.lower().startswith("https://")
    resp_headers: list[tuple[str, str]] = []
    for k, v in upstream_resp.headers.multi_items():
        lk = k.lower()
        if lk in _STRIPPED_RESPONSE_HEADERS:
            continue
        resp_headers.append((k, v))

    location = upstream_resp.headers.get("location")
    if location is not None:
        resp_headers.append(
            ("Location", rewrite_location_header(location, prefix=prefix, upstream_base_url=ui_base))
        )

    for raw_cookie in upstream_resp.headers.get_list("set-cookie"):
        scoped = filter_and_scope_set_cookie(raw_cookie, prefix=prefix, secure_cookie=secure_cookie)
        if scoped is not None:
            resp_headers.append(("Set-Cookie", scoped))

    async def _stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()

    response = StreamingResponse(
        _stream_body(),
        status_code=upstream_resp.status_code,
    )
    # Apply multi-value headers (including multiple Set-Cookie headers) cleanly
    for hk, hv in resp_headers:
        response.headers.append(hk, hv)
    return response
