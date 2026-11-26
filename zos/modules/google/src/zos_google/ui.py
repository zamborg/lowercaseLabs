"""Owner UI for ZOS Google modules (`/admin/m/gmail` and `/admin/m/gcal`).

Displays connection status only (never tokens or secrets), runs per-service Google OAuth
(`/connect` -> Google -> `/callback`), allows Disconnect, and exposes the `api|mcp`
backend switch for `gcal`.
"""

from __future__ import annotations

import hmac
import html
import secrets
from typing import Any
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from zos_google.calendar import GoogleCalendarService
from zos_google.caller import CallerVerificationError, verify_http_caller
from zos_google.oauth import GoogleClient

CSRF_COOKIE = "google_csrf"

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f5f6; --surface: #ffffff; --sunken: #f0f0f2;
  --fg: #18181b; --muted: #5a5a66; --border: #e4e4e7; --border-strong: #cfcfd6;
  --accent: #18181b; --accent-fg: #ffffff; --focus: #2563eb;
  --ok: #166534; --ok-bg: #dcfce7; --ok-bd: #86efac;
  --warn: #92400e; --warn-bg: #fef3c7; --warn-bd: #fcd34d;
  --err: #991b1b; --err-bg: #fee2e2; --err-bd: #fca5a5;
  --s1: 0.25rem; --s2: 0.5rem; --s3: 0.75rem; --s4: 1rem; --s5: 1.5rem; --s6: 2.25rem;
  --r-sm: 6px; --r: 10px;
  --fs-xs: 0.75rem; --fs-sm: 0.8125rem; --fs: 0.9375rem; --fs-lg: 1.0625rem; --fs-xl: 1.375rem;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --shadow: 0 1px 2px rgba(0,0,0,.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e0e10; --surface: #17171a; --sunken: #121215;
    --fg: #f4f4f5; --muted: #a1a1ab; --border: #2a2a30; --border-strong: #3d3d46;
    --accent: #f4f4f5; --accent-fg: #18181b; --focus: #60a5fa;
    --ok: #86efac; --ok-bg: #052e16; --ok-bd: #166534;
    --warn: #fcd34d; --warn-bg: #3a2105; --warn-bd: #92400e;
    --err: #fca5a5; --err-bg: #3f0d0d; --err-bd: #991b1b;
    --shadow: none;
  }
}
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; font-family: var(--sans); font-size: var(--fs); line-height: 1.5; background: var(--bg); color: var(--fg); overflow-wrap: anywhere; }
a { color: inherit; }
h1 { font-size: var(--fs-xl); margin: 0 0 var(--s3); }
h2 { font-size: var(--fs-lg); margin: 0 0 var(--s3); }
code { font-family: var(--mono); font-size: 0.85em; background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 0.05em 0.35em; }
.topbar { background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 20; }
.topbar-in { max-width: 960px; margin: 0 auto; padding: var(--s2) var(--s4); display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; }
.brand { font-weight: 700; text-decoration: none; display: inline-flex; align-items: center; gap: var(--s2); }
.brand::before { content: ""; width: 0.65rem; height: 0.65rem; border-radius: 3px; background: var(--accent); }
main { max-width: 960px; margin: 0 auto; padding: var(--s4) var(--s4) var(--s6); }
.row { display: flex; gap: var(--s2); flex-wrap: wrap; align-items: center; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s4); margin-bottom: var(--s4); box-shadow: var(--shadow); }
.muted { color: var(--muted); font-size: var(--fs-sm); }
.badge { display: inline-flex; align-items: center; gap: 0.4em; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: var(--fs-xs); font-weight: 600; border: 1px solid transparent; }
.badge::before { content: ""; width: 0.45em; height: 0.45em; border-radius: 50%; background: currentColor; }
.badge-ok { background: var(--ok-bg); color: var(--ok); border-color: var(--ok-bd); }
.badge-warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-bd); }
.notice { padding: var(--s3) var(--s4); border-radius: var(--r); margin-bottom: var(--s4); border: 1px solid; }
.notice-ok { background: var(--ok-bg); border-color: var(--ok-bd); color: var(--ok); }
.notice-err { background: var(--err-bg); border-color: var(--err-bd); color: var(--err); }
form { margin: 0; }
select { font: inherit; color: var(--fg); background: var(--surface); min-height: 40px; padding: 0 var(--s3); border: 1px solid var(--border-strong); border-radius: var(--r-sm); }
button, a.btn { display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); text-decoration: none; font: inherit; font-weight: 600; min-height: 40px; padding: 0 var(--s4); border-radius: var(--r-sm); border: 1px solid var(--accent); background: var(--accent); color: var(--accent-fg); cursor: pointer; }
button.secondary, a.btn.secondary { background: var(--surface); color: var(--fg); border-color: var(--border-strong); }
button.danger { background: var(--surface); color: var(--err); border-color: var(--err-bd); }
button.sm, a.btn.sm { min-height: 36px; padding: 0 var(--s3); font-size: var(--fs-sm); }
dl.kv { margin: 0 0 var(--s4); }
dl.kv dt { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; margin-top: var(--s3); }
dl.kv dd { margin: var(--s1) 0 0; }
.chips { display: flex; gap: var(--s1); flex-wrap: wrap; }
.chip { font-size: var(--fs-xs); padding: 0.1rem 0.5rem; background: var(--sunken); border: 1px solid var(--border); border-radius: 999px; font-family: var(--mono); }
"""


def _e(val: Any) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def _prefix(request: Request, *, service: str) -> str:
    raw = request.headers.get("x-forwarded-prefix", "").strip().rstrip("/")
    if raw.startswith("/"):
        return raw
    return f"/admin/m/{service}"


def _derive_callback_url(request: Request, *, service: str) -> str:
    proto = request.headers.get("x-forwarded-proto", "").strip().lower()
    if proto not in ("http", "https"):
        proto = request.url.scheme or "http"
    host = request.headers.get("x-forwarded-host", "").strip() or request.headers.get("host", "").strip()
    if not host:
        host = request.url.netloc
    prefix = _prefix(request, service=service)
    return f"{proto}://{host}{prefix}/callback"


def _ensure_csrf(request: Request) -> tuple[str, bool]:
    existing = request.cookies.get(CSRF_COOKIE, "").strip()
    if existing:
        return existing, False
    return secrets.token_urlsafe(24), True


def _attach_csrf_cookie(response: Response, token: str, *, prefix: str, is_new: bool) -> Response:
    if is_new:
        response.set_cookie(
            key=CSRF_COOKIE,
            value=token,
            path=prefix or "/",
            httponly=True,
            samesite="strict",
        )
    return response


def _check_csrf(request: Request, form_token: str) -> bool:
    cookie_csrf = request.cookies.get(CSRF_COOKIE, "").strip()
    return bool(cookie_csrf and form_token and hmac.compare_digest(cookie_csrf, form_token))


def create_ui_app(
    client: GoogleClient,
    *,
    service: str,
    module_secret: str,
    calendar_service: GoogleCalendarService | None = None,
) -> Starlette:
    """Create the owner UI Starlette app for `gmail` or `gcal`."""
    title_label = "Gmail" if service == "gmail" else "Google Calendar"

    def _verify_owner(request: Request) -> str | None:
        try:
            caller = verify_http_caller(dict(request.headers), secret=module_secret)
        except CallerVerificationError:
            return None
        return caller.client_name

    def _page(body: str, *, prefix: str) -> str:
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{_e(title_label)} — ZOS</title>
  <style>{_CSS}</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-in">
      <div class="row">
        <a href="/admin" class="btn secondary sm">&larr; Admin</a>
        <a class="brand" href="{_e(prefix)}/">{_e(title_label)}</a>
      </div>
      <span class="muted">module: <code>{_e(service)}</code></span>
    </div>
  </header>
  <main>{body}</main>
</body>
</html>"""

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "module": service, "service": "ui"})

    async def index_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request, service=service)
        csrf_token, is_new = _ensure_csrf(request)
        status = await client.store.get_connection_status()
        callback_url = _derive_callback_url(request, service=service)

        notice = request.query_params.get("notice", "").strip()
        err = request.query_params.get("error", "").strip()
        notice_html = f'<div class="notice notice-ok">{_e(notice)}</div>' if notice else ""
        err_html = f'<div class="notice notice-err">{_e(err)}</div>' if err else ""

        connected = bool(status["connected"])
        badge = (
            '<span class="badge badge-ok">Connected</span>'
            if connected
            else '<span class="badge badge-warn">Not connected</span>'
        )
        scopes_html = (
            "".join(f'<span class="chip">{_e(s)}</span>' for s in (status["scopes"] or client.scopes))
        )

        connect_label = f"Reconnect {title_label}" if connected else f"Connect {title_label}"
        connect_disabled = "" if client.is_configured else "disabled"
        config_warn = (
            ""
            if client.is_configured
            else '<div class="notice notice-err">GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are not configured for this module.</div>'
        )

        disconnect_form = ""
        if connected:
            disconnect_form = f"""
            <form method="post" action="{_e(pref)}/disconnect" onsubmit="return confirm('Disconnect {title_label} and delete stored tokens?');">
              <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
              <button type="submit" class="danger">Disconnect</button>
            </form>
            """

        backend_card = ""
        if service == "gcal" and calendar_service is not None:
            cur_backend = await calendar_service.get_backend()
            backend_card = f"""
            <div class="card">
              <h2>Calendar Backend</h2>
              <p class="muted">Switch between Google Calendar REST API (<code>api</code>) and Google's remote Calendar MCP server (<code>mcp</code>).</p>
              <form method="post" action="{_e(pref)}/backend" class="row">
                <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
                <select name="backend">
                  <option value="api" {"selected" if cur_backend == "api" else ""}>api (REST v3)</option>
                  <option value="mcp" {"selected" if cur_backend == "mcp" else ""}>mcp (Google Calendar MCP)</option>
                </select>
                <button type="submit" class="secondary">Save Backend</button>
              </form>
            </div>
            """

        account_display = _e(status["account"]) if connected else '<span class="muted">—</span>'
        updated_display = (
            _e(str(status["updated_at"]).replace("T", " ")[:19] + " UTC")
            if status.get("updated_at")
            else '<span class="muted">—</span>'
        )

        body = f"""
        {notice_html}
        {err_html}
        {config_warn}
        <div class="card">
          <div class="row" style="justify-content: space-between;">
            <h1>{_e(title_label)}</h1>
            {badge}
          </div>
          <dl class="kv">
            <dt>Account</dt>
            <dd>{account_display}</dd>
            <dt>Last Updated</dt>
            <dd>{updated_display}</dd>
            <dt>OAuth Redirect URI</dt>
            <dd><code>{_e(callback_url)}</code></dd>
            <dt>Scopes</dt>
            <dd><div class="chips">{scopes_html}</div></dd>
          </dl>
          <div class="row">
            <form method="post" action="{_e(pref)}/connect">
              <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
              <button type="submit" {connect_disabled}>{_e(connect_label)}</button>
            </form>
            {disconnect_form}
          </div>
        </div>
        {backend_card}
        """
        resp = HTMLResponse(_page(body, prefix=pref), headers={"Cache-Control": "no-store"})
        return _attach_csrf_cookie(resp, csrf_token, prefix=pref, is_new=is_new)

    async def connect_action(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request, service=service)
        form = await request.form()
        if not _check_csrf(request, str(form.get("csrf_token", "")).strip()):
            return HTMLResponse("<h1>Invalid CSRF token</h1>", status_code=403)
        if not client.is_configured:
            return RedirectResponse(
                f"{pref}/?error={quote('GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured.')}",
                status_code=303,
            )
        state = await client.store.create_oauth_state()
        callback_url = _derive_callback_url(request, service=service)
        auth_url = client.build_authorize_url(redirect_uri=callback_url, state=state)
        return RedirectResponse(auth_url, status_code=302)

    async def callback_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request, service=service)
        error = request.query_params.get("error", "").strip()
        if error:
            return RedirectResponse(
                f"{pref}/?error={quote(f'Google authorization failed: {error}')}",
                status_code=303,
            )
        state = request.query_params.get("state", "").strip()
        code = request.query_params.get("code", "").strip()
        if not code:
            return RedirectResponse(
                f"{pref}/?error={quote('Missing authorization code.')}",
                status_code=303,
            )
        if not await client.store.consume_oauth_state(state):
            return RedirectResponse(
                f"{pref}/?error={quote('Invalid or expired OAuth state.')}",
                status_code=303,
            )
        callback_url = _derive_callback_url(request, service=service)
        try:
            await client.exchange_code(code=code, redirect_uri=callback_url)
        except Exception as exc:
            return RedirectResponse(
                f"{pref}/?error={quote(f'OAuth exchange failed: {exc}')}",
                status_code=303,
            )
        return RedirectResponse(
            f"{pref}/?notice={quote(f'{title_label} connected.')}",
            status_code=303,
        )

    async def disconnect_action(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request, service=service)
        form = await request.form()
        if not _check_csrf(request, str(form.get("csrf_token", "")).strip()):
            return HTMLResponse("<h1>Invalid CSRF token</h1>", status_code=403)
        await client.store.delete_tokens()
        return RedirectResponse(
            f"{pref}/?notice={quote(f'{title_label} disconnected.')}",
            status_code=303,
        )

    async def backend_action(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request, service=service)
        form = await request.form()
        if not _check_csrf(request, str(form.get("csrf_token", "")).strip()):
            return HTMLResponse("<h1>Invalid CSRF token</h1>", status_code=403)
        if calendar_service is None:
            return HTMLResponse("<h1>Not supported</h1>", status_code=404)
        backend = str(form.get("backend", "")).strip().lower()
        try:
            saved = await calendar_service.set_backend(backend)
        except ValueError as exc:
            return RedirectResponse(f"{pref}/?error={quote(str(exc))}", status_code=303)
        return RedirectResponse(
            f"{pref}/?notice={quote(f'Calendar backend set to {saved}.')}",
            status_code=303,
        )

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/", index_view, methods=["GET"]),
            Route("/connect", connect_action, methods=["POST"]),
            Route("/callback", callback_view, methods=["GET"]),
            Route("/disconnect", disconnect_action, methods=["POST"]),
            Route("/backend", backend_action, methods=["POST"]),
        ]
    )
