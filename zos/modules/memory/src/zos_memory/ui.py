"""Mobile-first server-rendered owner UI for the ZOS Memory module (browse, search, delete)."""

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

from zos_memory.caller import CallerVerificationError, verify_http_caller
from zos_memory.store import MemoryStore

CSRF_COOKIE = "memory_csrf"

# Design tokens/components mirror zos/web/theme.py (inlined: this module must not import zos).
_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f5f6; --surface: #ffffff; --surface-2: #fafafa; --sunken: #f0f0f2;
  --fg: #18181b; --muted: #5a5a66; --border: #e4e4e7; --border-strong: #cfcfd6;
  --accent: #18181b; --accent-fg: #ffffff; --focus: #2563eb;
  --ok: #166534; --ok-bg: #dcfce7; --ok-bd: #86efac;
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
    --bg: #0e0e10; --surface: #17171a; --surface-2: #1c1c20; --sunken: #121215;
    --fg: #f4f4f5; --muted: #a1a1ab; --border: #2a2a30; --border-strong: #3d3d46;
    --accent: #f4f4f5; --accent-fg: #18181b; --focus: #60a5fa;
    --ok: #86efac; --ok-bg: #052e16; --ok-bd: #166534;
    --err: #fca5a5; --err-bg: #3f0d0d; --err-bd: #991b1b;
    --shadow: none;
  }
}
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; font-family: var(--sans); font-size: var(--fs); line-height: 1.5; background: var(--bg); color: var(--fg); overflow-wrap: anywhere; }
a { color: inherit; text-underline-offset: 2px; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: var(--r-sm); }
h1 { font-size: var(--fs-xl); line-height: 1.25; margin: 0 0 var(--s4); letter-spacing: -0.01em; }
h2 { font-size: var(--fs-lg); line-height: 1.3; margin: 0 0 var(--s3); }
p { margin: 0 0 var(--s3); }
code, .mono { font-family: var(--mono); font-size: 0.85em; }
code { background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 0.05em 0.35em; }
.skip { position: absolute; left: -999px; top: 0; background: var(--accent); color: var(--accent-fg); padding: var(--s2) var(--s3); z-index: 100; }
.skip:focus { left: var(--s2); top: var(--s2); }
.topbar { background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 20; }
.topbar-in { max-width: 1180px; margin: 0 auto; padding: var(--s2) var(--s4); display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; }
.brand { font-weight: 700; letter-spacing: 0.01em; text-decoration: none; padding: var(--s2) 0; display: inline-flex; align-items: center; gap: var(--s2); }
.brand::before { content: ""; width: 0.65rem; height: 0.65rem; border-radius: 3px; background: var(--accent); }
main { max-width: 1180px; margin: 0 auto; padding: var(--s4) var(--s4) var(--s6); }
.row { display: flex; gap: var(--s2); flex-wrap: wrap; align-items: center; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s4); margin-bottom: var(--s4); box-shadow: var(--shadow); }
.card > :last-child { margin-bottom: 0; }
.muted { color: var(--muted); font-size: var(--fs-sm); }
.empty { text-align: center; color: var(--muted); padding: var(--s5) var(--s3); }
.notice { padding: var(--s3) var(--s4); border-radius: var(--r); margin-bottom: var(--s4); border: 1px solid; }
.notice-ok { background: var(--ok-bg); border-color: var(--ok-bd); color: var(--ok); }
.notice-err { background: var(--err-bg); border-color: var(--err-bd); color: var(--err); }
form { margin: 0; }
.search-bar { display: flex; gap: var(--s2); flex-wrap: wrap; margin-bottom: var(--s3); }
input[type=text], input[type=search] { font: inherit; font-size: var(--fs); color: var(--fg); background: var(--surface); padding: 0 var(--s3); min-height: 44px; border: 1px solid var(--border-strong); border-radius: var(--r-sm); flex: 1 1 16rem; max-width: 100%; }
button, a.btn { display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); text-decoration: none; font: inherit; font-size: var(--fs); font-weight: 600; min-height: 44px; padding: 0 var(--s4); border-radius: var(--r-sm); border: 1px solid var(--accent); background: var(--accent); color: var(--accent-fg); cursor: pointer; white-space: nowrap; }
button:hover, a.btn:hover { opacity: 0.88; }
button.secondary, a.btn.secondary { background: var(--surface); color: var(--fg); border-color: var(--border-strong); }
button.secondary:hover, a.btn.secondary:hover { background: var(--sunken); opacity: 1; }
button.danger { background: var(--surface); color: var(--err); border-color: var(--err-bd); }
button.danger:hover { background: var(--err-bg); opacity: 1; }
button.sm, a.btn.sm { min-height: 38px; padding: 0 var(--s3); font-size: var(--fs-sm); }
.pills { display: flex; gap: var(--s2); overflow-x: auto; padding-bottom: var(--s2); margin-bottom: var(--s3); scrollbar-width: none; }
.pills::-webkit-scrollbar { display: none; }
.pill { display: inline-flex; align-items: center; gap: var(--s2); white-space: nowrap; text-decoration: none; min-height: 36px; padding: 0 var(--s3); border-radius: 999px; background: var(--surface); border: 1px solid var(--border-strong); font-size: var(--fs-sm); }
.pill.active { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); font-weight: 600; }
.mem-list { display: flex; flex-direction: column; gap: var(--s2); }
.mem-row { display: block; text-decoration: none; color: inherit; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s3) var(--s4); box-shadow: var(--shadow); }
.mem-row:hover { border-color: var(--border-strong); }
.mem-row .top { display: flex; justify-content: space-between; gap: var(--s3); align-items: baseline; flex-wrap: wrap; }
.mem-row .key { font-family: var(--mono); font-weight: 600; font-size: var(--fs); }
.mem-row .meta { font-size: var(--fs-sm); color: var(--muted); margin-top: var(--s1); display: flex; gap: var(--s2); flex-wrap: wrap; align-items: center; }
.mem-row .snippet { font-size: var(--fs-sm); color: var(--muted); margin-top: var(--s2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chips { display: flex; gap: var(--s1); flex-wrap: wrap; }
.chip { font-size: var(--fs-xs); padding: 0.1rem 0.5rem; background: var(--sunken); border: 1px solid var(--border); border-radius: 999px; font-family: var(--mono); }
.mem-body { white-space: pre-wrap; word-break: break-word; background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--r-sm); padding: var(--s4); font-family: var(--mono); font-size: var(--fs-sm); line-height: 1.6; }
.back { margin: 0 0 var(--s3); }
@media (max-width: 720px) {
  main { padding: var(--s3) var(--s3) var(--s5); }
  input[type=text], input[type=search] { font-size: 16px; }
}
"""


def _e(val: Any) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def _prefix(request: Request) -> str:
    raw = request.headers.get("x-forwarded-prefix", "").strip().rstrip("/")
    return raw if raw.startswith("/") else ""


def _fmt_time(iso_str: str | None) -> str:
    if not iso_str:
        return ""
    return iso_str.replace("T", " ")[:16] + " UTC"


def _page(*, title: str, prefix: str, body: str, total: int | None = None) -> str:
    count_badge = f'<span class="muted">{total} memories</span>' if total is not None else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{_e(title)} — ZOS Memory</title>
  <style>{_CSS}</style>
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <header class="topbar">
    <div class="topbar-in">
      <div class="row">
        <a href="/admin" class="btn secondary sm">&larr; Admin</a>
        <a class="brand" href="{_e(prefix)}/">Memory</a>
      </div>
      {count_badge}
    </div>
  </header>
  <main id="main">{body}</main>
</body>
</html>"""


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


def create_ui_app(store: MemoryStore, *, module_secret: str) -> Starlette:
    """Create the owner UI for browsing, searching, and deleting memories (no editing)."""

    def _verify_owner(request: Request) -> str | None:
        try:
            caller = verify_http_caller(dict(request.headers), secret=module_secret)
        except CallerVerificationError:
            return None
        return caller.client_name

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "module": "memory", "service": "ui"})

    async def index_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        csrf_token, is_new = _ensure_csrf(request)

        query = request.query_params.get("q", "").strip()
        selected_tag = request.query_params.get("tag", "").strip()
        notice = request.query_params.get("notice", "").strip()

        items, total_count, tag_counts = await store.browse_for_ui(
            query=query,
            tag=selected_tag,
            limit=100,
        )

        notice_html = f'<div class="notice notice-ok">{_e(notice)}</div>' if notice else ""

        tag_pills = [
            f'<a class="pill {"active" if not selected_tag else ""}" href="{_e(pref)}/'
            + (f"?q={quote(query)}" if query else "")
            + '">All tags</a>'
        ]
        for tname, tcnt in tag_counts:
            qs = f"tag={quote(tname)}" + (f"&q={quote(query)}" if query else "")
            tag_pills.append(
                f'<a class="pill {"active" if selected_tag == tname else ""}" '
                f'href="{_e(pref)}/?{qs}">{_e(tname)} <span class="muted">({tcnt})</span></a>'
            )

        rows_html: list[str] = []
        for item in items:
            snippet = item["content"].replace("\n", " ").strip()[:140]
            author = item.get("author") or "unknown"
            chips = "".join(f'<span class="chip">{_e(t)}</span>' for t in item["tags"])
            rows_html.append(
                f'<a class="mem-row" href="{_e(pref)}/item/{_e(item["id"])}">'
                f'<div class="top"><span class="key">{_e(item["key"])}</span>'
                f'<span class="muted">{_e(_fmt_time(item["updated_at"]))}</span></div>'
                f'<div class="meta"><span>Client: <strong>{_e(author)}</strong></span>'
                f'<span class="chips">{chips}</span></div>'
                f'<div class="snippet">{_e(snippet)}</div>'
                f'</a>'
            )

        list_html = (
            f'<div class="mem-list">{"".join(rows_html)}</div>'
            if rows_html
            else '<div class="card"><p class="empty">No matching memories.</p></div>'
        )

        clear_btn = (
            f'<a class="btn secondary" href="{_e(pref)}/">Clear</a>'
            if (query or selected_tag)
            else ""
        )
        tag_hidden = (
            f'<input type="hidden" name="tag" value="{_e(selected_tag)}">' if selected_tag else ""
        )

        body = f"""
        {notice_html}
        <form class="search-bar" method="get" action="{_e(pref)}/">
          {tag_hidden}
          <input type="search" name="q" value="{_e(query)}" placeholder="Search memories (full-text or fuzzy)...">
          <button type="submit">Search</button>
          {clear_btn}
        </form>
        <div class="pills">{"".join(tag_pills)}</div>
        {list_html}
        """
        resp = HTMLResponse(_page(title="Memories", prefix=pref, body=body, total=total_count))
        return _attach_csrf_cookie(resp, csrf_token, prefix=pref, is_new=is_new)

    async def detail_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        csrf_token, is_new = _ensure_csrf(request)
        memory_id = str(request.path_params.get("memory_id", "")).strip()

        try:
            item = await store.get_memory_by_id(memory_id=memory_id)
        except ValueError:
            item = None

        if item is None:
            return HTMLResponse(
                _page(title="Not Found", prefix=pref, body='<div class="card"><p>Memory not found.</p></div>'),
                status_code=404,
            )

        author = item.get("author") or "unknown"
        chips = "".join(f'<span class="chip">{_e(t)}</span>' for t in item["tags"]) or '<span class="muted">none</span>'
        body = f"""
        <p class="back"><a class="btn secondary sm" href="{_e(pref)}/">&larr; All Memories</a></p>
        <div class="card">
          <h1><code>{_e(item["key"])}</code></h1>
          <div class="row" style="margin-bottom: var(--s3);">
            <span class="muted">Written by <strong>{_e(author)}</strong></span>
            <span class="muted">&middot; Created {_e(_fmt_time(item["created_at"]))}</span>
            <span class="muted">&middot; Updated {_e(_fmt_time(item["updated_at"]))}</span>
          </div>
          <div class="chips" style="margin-bottom: var(--s4);">{chips}</div>
          <div class="mem-body">{_e(item["content"])}</div>
          <div style="margin-top: var(--s4);">
            <form method="post" action="{_e(pref)}/delete" onsubmit="return confirm('Delete this memory permanently?');">
              <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
              <input type="hidden" name="key" value="{_e(item["key"])}">
              <button type="submit" class="danger sm">Delete Memory</button>
            </form>
          </div>
        </div>
        """
        resp = HTMLResponse(_page(title=item["key"], prefix=pref, body=body))
        return _attach_csrf_cookie(resp, csrf_token, prefix=pref, is_new=is_new)

    async def delete_action(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        form = await request.form()
        cookie_csrf = request.cookies.get(CSRF_COOKIE, "").strip()
        form_csrf = str(form.get("csrf_token", "")).strip()
        if not cookie_csrf or not form_csrf or not hmac.compare_digest(cookie_csrf, form_csrf):
            return HTMLResponse("<h1>Invalid CSRF token</h1>", status_code=403)

        key = str(form.get("key", "")).strip()
        if key:
            await store.delete_memory(key=key)
        return RedirectResponse(f"{pref}/?notice={quote('Memory deleted.')}", status_code=303)

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/", index_view, methods=["GET"]),
            Route("/item/{memory_id:str}", detail_view, methods=["GET"]),
            Route("/delete", delete_action, methods=["POST"]),
        ]
    )
