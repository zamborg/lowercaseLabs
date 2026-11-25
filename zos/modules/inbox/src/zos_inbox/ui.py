"""Mobile-first server-rendered owner UI for the ZOS Inbox module."""

from __future__ import annotations

import contextlib
import hmac
import html
import secrets
from typing import Any
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from zos_inbox.caller import CallerVerificationError, verify_http_caller
from zos_inbox.store import InboxStore

CSRF_COOKIE = "inbox_csrf"

# Design tokens/components below mirror zos/web/theme.py (inlined: this module must not import zos).
_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f5f6; --surface: #ffffff; --surface-2: #fafafa; --sunken: #f0f0f2;
  --fg: #18181b; --muted: #5a5a66; --border: #e4e4e7; --border-strong: #cfcfd6;
  --accent: #18181b; --accent-fg: #ffffff; --focus: #2563eb;
  --ok: #166534; --ok-bg: #dcfce7; --ok-bd: #86efac;
  --warn: #92400e; --warn-bg: #fef3c7; --warn-bd: #fcd34d;
  --err: #991b1b; --err-bg: #fee2e2; --err-bd: #fca5a5;
  --info: #1e40af; --info-bg: #dbeafe;
  --code-bg: #18181b; --code-fg: #f4f4f5;
  --s1: 0.25rem; --s2: 0.5rem; --s3: 0.75rem; --s4: 1rem; --s5: 1.5rem; --s6: 2.25rem;
  --r-sm: 6px; --r: 10px; --r-lg: 14px;
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
    --warn: #fcd34d; --warn-bg: #3a2105; --warn-bd: #92400e;
    --err: #fca5a5; --err-bg: #3f0d0d; --err-bd: #991b1b;
    --info: #93c5fd; --info-bg: #172554;
    --code-bg: #0a0a0c; --code-fg: #f4f4f5;
    --shadow: none;
  }
}
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; font-family: var(--sans); font-size: var(--fs); line-height: 1.5; background: var(--bg); color: var(--fg); overflow-wrap: anywhere; }
a { color: inherit; text-underline-offset: 2px; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: var(--r-sm); }
h1 { font-size: var(--fs-xl); line-height: 1.25; margin: 0 0 var(--s4); letter-spacing: -0.01em; }
h2 { font-size: var(--fs-lg); line-height: 1.3; margin: 0 0 var(--s3); display: flex; align-items: center; gap: var(--s2); flex-wrap: wrap; }
p { margin: 0 0 var(--s3); }
code, .mono { font-family: var(--mono); font-size: 0.85em; }
code { background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 0.05em 0.35em; }
td code, th code, .badge code, pre code, .token-box code { background: none; border: 0; padding: 0; }
.skip { position: absolute; left: -999px; top: 0; background: var(--accent); color: var(--accent-fg); padding: var(--s2) var(--s3); z-index: 100; }
.skip:focus { left: var(--s2); top: var(--s2); }

/* Shell: header + scrollable nav */
.topbar { background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 20; }
.topbar-in { max-width: 1180px; margin: 0 auto; padding: var(--s2) var(--s4) 0; display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; }
.brand { font-weight: 700; letter-spacing: 0.01em; text-decoration: none; padding: var(--s2) 0; display: inline-flex; align-items: center; gap: var(--s2); }
.brand::before { content: ""; width: 0.65rem; height: 0.65rem; border-radius: 3px; background: var(--accent); }
nav.main-nav { display: flex; gap: var(--s1); overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch; max-width: 1180px; margin: 0 auto; padding: 0 var(--s3); }
nav.main-nav::-webkit-scrollbar { display: none; }
nav.main-nav a { text-decoration: none; white-space: nowrap; color: var(--muted); padding: 0 var(--s3); min-height: 44px; display: inline-flex; align-items: center; gap: var(--s2); border-bottom: 2px solid transparent; font-size: var(--fs); font-weight: 500; }
nav.main-nav a:hover { color: var(--fg); }
nav.main-nav a.active { color: var(--fg); border-bottom-color: var(--accent); font-weight: 600; }
main { max-width: 1180px; margin: 0 auto; padding: var(--s4) var(--s4) var(--s6); }
main.narrow { max-width: 30rem; padding-top: var(--s5); }
.page-head { display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; margin-bottom: var(--s4); }
.page-head h1 { margin: 0; }
.back { margin: 0 0 var(--s3); }
.pager { margin: var(--s4) 0 0; }

/* Surfaces */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s4); margin-bottom: var(--s4); box-shadow: var(--shadow); }
.card > :last-child { margin-bottom: 0; }
.stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s3); margin-bottom: var(--s4); }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s3) var(--s4); box-shadow: var(--shadow); min-width: 0; }
.stat .label, .meta-item .label { font-size: var(--fs-xs); color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.stat .value { font-size: 1.5rem; font-weight: 700; margin-top: var(--s1); line-height: 1.2; }
.stat .value .sub { display: block; font-size: var(--fs-sm); font-weight: 400; color: var(--muted); margin-top: var(--s1); }
.muted { color: var(--muted); font-size: var(--fs-sm); }
.muted.block { display: block; margin-top: var(--s1); }
.err-text { color: var(--err); }
.empty { text-align: center; color: var(--muted); padding: var(--s5) var(--s3); font-size: var(--fs); }
.row { display: flex; gap: var(--s2); flex-wrap: wrap; align-items: center; }
.row.gap-lg { gap: var(--s3); }
.stack-sm > * + * { margin-top: var(--s2); }

/* Tables: scroll inside container; .stack tables become cards on narrow screens */
.scroll-x { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0 calc(-1 * var(--s1)); padding: 0 var(--s1); }
table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
th, td { text-align: left; padding: var(--s2) var(--s3); border-bottom: 1px solid var(--border); vertical-align: middle; }
th { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.nowrap { white-space: nowrap; }
.digest { font-family: var(--mono); font-size: var(--fs-xs); color: var(--muted); }

/* Status pills */
.badge { display: inline-flex; align-items: center; gap: 0.4em; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: var(--fs-xs); font-weight: 600; line-height: 1.4; white-space: nowrap; border: 1px solid transparent; }
.badge::before { content: ""; width: 0.45em; height: 0.45em; border-radius: 50%; background: currentColor; opacity: 0.85; }
.badge-ok { background: var(--ok-bg); color: var(--ok); border-color: var(--ok-bd); }
.badge-warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-bd); }
.badge-err { background: var(--err-bg); color: var(--err); border-color: var(--err-bd); }
.badge-muted { background: var(--sunken); color: var(--muted); border-color: var(--border); }
.badge-info { background: var(--info-bg); color: var(--info); }
.badge.plain::before { display: none; }

/* Notices */
.notice { padding: var(--s3) var(--s4); border-radius: var(--r); margin-bottom: var(--s4); border: 1px solid; font-size: var(--fs); }
.notice-ok { background: var(--ok-bg); border-color: var(--ok-bd); color: var(--ok); }
.notice-err { background: var(--err-bg); border-color: var(--err-bd); color: var(--err); }
.notice-token { background: var(--warn-bg); border-color: var(--warn-bd); color: var(--warn); }
.token-box { margin-top: var(--s2); padding: var(--s3); background: var(--surface); color: var(--fg); border: 1px dashed var(--warn-bd); border-radius: var(--r-sm); font-family: var(--mono); word-break: break-all; font-size: var(--fs-sm); user-select: all; }

/* Forms and buttons */
form { margin: 0; }
form.inline { display: inline-flex; gap: var(--s2); align-items: center; flex-wrap: wrap; }
.form-row { display: flex; gap: var(--s3); flex-wrap: wrap; align-items: flex-end; }
.form-row > .grow { flex: 1 1 14rem; }
label.field { display: flex; flex-direction: column; gap: var(--s1); font-size: var(--fs-sm); font-weight: 600; min-width: 0; }
label.field.block { margin-bottom: var(--s4); }
input[type=text], input[type=password], select, textarea { font: inherit; font-size: var(--fs); color: var(--fg); background: var(--surface); padding: 0 var(--s3); min-height: 44px; border: 1px solid var(--border-strong); border-radius: var(--r-sm); max-width: 100%; }
input[type=text], input[type=password], textarea { width: 100%; }
td input[type=text], td input[type=password], .row input[type=text], .row input[type=password], form.inline input[type=text], form.inline input[type=password] { width: 11rem; }
input::placeholder, textarea::placeholder { color: var(--muted); opacity: 0.8; }
textarea { padding: var(--s3); min-height: 9rem; resize: vertical; line-height: 1.5; }
select { padding-right: var(--s2); }
button, a.btn { display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); text-decoration: none; font: inherit; font-size: var(--fs); font-weight: 600; min-height: 44px; padding: 0 var(--s4); border-radius: var(--r-sm); border: 1px solid var(--accent); background: var(--accent); color: var(--accent-fg); cursor: pointer; white-space: nowrap; }
button:hover, a.btn:hover { opacity: 0.88; }
button.secondary, a.btn.secondary { background: var(--surface); color: var(--fg); border-color: var(--border-strong); }
button.secondary:hover, a.btn.secondary:hover { background: var(--sunken); opacity: 1; }
button.danger { background: var(--surface); color: var(--err); border-color: var(--err-bd); }
button.danger:hover { background: var(--err-bg); opacity: 1; }
button.sm, a.btn.sm { min-height: 44px; padding: 0 var(--s3); font-size: var(--fs-sm); }
button.block, .block-btn { width: 100%; }
@media (min-width: 721px) and (pointer: fine) {
  button.sm, a.btn.sm { min-height: 34px; padding: 0 var(--s3); }
  input[type=text], input[type=password], select { min-height: 38px; }
  button, a.btn { min-height: 38px; }
  nav.main-nav a { min-height: 44px; }
}
.cell-controls { display: flex; flex-direction: column; gap: var(--s2); align-items: flex-start; }

/* Approval detail */
.approval-pre, pre.code { background: var(--code-bg); color: var(--code-fg); padding: var(--s4); border-radius: var(--r-sm); font-family: var(--mono); font-size: var(--fs-sm); line-height: 1.55; white-space: pre-wrap; word-break: break-word; overflow-x: auto; margin: var(--s2) 0 0; border: 1px solid var(--border); }
.approval-actions { display: grid; grid-template-columns: 1fr 1fr; gap: var(--s3); margin-top: var(--s4); }
.approval-actions button { width: 100%; min-height: 52px; font-size: var(--fs-lg); border-radius: var(--r); }
button.approve-btn { background: var(--ok); border-color: var(--ok); color: var(--surface); }
button.deny-btn { background: var(--err); border-color: var(--err); color: var(--surface); }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: var(--s4) var(--s3); }
.meta-item .val { margin-top: var(--s1); word-break: break-word; }

/* Key/value list (authorize + connect pages) */
dl.kv { margin: 0 0 var(--s4); }
dl.kv dt { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; margin-top: var(--s3); }
dl.kv dd { margin: var(--s1) 0 0; font-family: var(--mono); font-size: var(--fs-sm); word-break: break-all; }
dd.host-callout { background: var(--warn-bg); border: 1px solid var(--warn-bd); color: var(--warn); border-radius: var(--r-sm); padding: var(--s2) var(--s3); font-size: var(--fs); font-weight: 700; }
.meta-box { background: var(--sunken); border: 1px solid var(--border); border-radius: var(--r-sm); padding: var(--s3); font-family: var(--mono); font-size: var(--fs-sm); word-break: break-all; margin: var(--s3) 0 var(--s4); }
.auth-actions { display: grid; grid-template-columns: 1fr 1fr; gap: var(--s3); }
.auth-form > * + * { margin-top: var(--s4); }

/* Narrow screens: stacked table rows */
@media (max-width: 720px) {
  main { padding: var(--s3) var(--s3) var(--s5); }
  .topbar-in { padding: 0 var(--s3); }
  nav.main-nav { padding: 0 var(--s2); }
  .card { padding: var(--s3) var(--s3); }
  input[type=text], input[type=password], select, textarea { font-size: 16px; }
  table.stack, table.stack tbody, table.stack tr, table.stack td { display: block; width: 100%; }
  table.stack thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
  table.stack tr { border: 1px solid var(--border); border-radius: var(--r); padding: var(--s2) var(--s3); margin-bottom: var(--s3); background: var(--surface-2); }
  table.stack td { border: 0; padding: var(--s1) 0; display: flex; gap: var(--s3); justify-content: space-between; align-items: center; text-align: right; }
  table.stack td::before { content: attr(data-label); color: var(--muted); font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; text-align: left; flex: none; }
  table.stack td.cell-wide { display: block; text-align: left; }
  table.stack td.cell-wide::before { display: block; margin-bottom: var(--s1); }
  table.stack td[colspan] { display: block; text-align: center; }
  table.stack td[colspan]::before { display: none; }
  form.inline { width: 100%; }
  form.inline input[type=text], form.inline input[type=password], td input[type=text], td input[type=password] { flex: 1 1 8rem; width: auto; min-width: 0; }
  .form-row > label.field, .form-row > button, .form-row > a.btn { flex: 1 1 100%; }
  .stat .value { font-size: 1.3rem; }
}
@media (min-width: 721px) {
  .stats { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  main { padding: var(--s5) var(--s5) var(--s6); }
  .topbar-in { padding: var(--s2) var(--s5) 0; }
}
@media (prefers-reduced-motion: no-preference) {
  button, a.btn, nav.main-nav a { transition: background-color .12s, opacity .12s, color .12s; }
}

/* Inbox-specific components */
.pills { display: flex; gap: var(--s2); overflow-x: auto; padding-bottom: var(--s2); margin-bottom: var(--s3); scrollbar-width: none; -webkit-overflow-scrolling: touch; }
.pills::-webkit-scrollbar { display: none; }
.pill { display: inline-flex; align-items: center; gap: var(--s2); white-space: nowrap; text-decoration: none; min-height: 44px; padding: 0 var(--s4); border-radius: 999px; background: var(--surface); border: 1px solid var(--border-strong); font-size: var(--fs); }
.pill.active { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); font-weight: 600; }
.pill .count { background: var(--focus); color: #fff; border-radius: 999px; padding: 0 0.5rem; font-size: var(--fs-xs); font-weight: 700; line-height: 1.5; }
.pill.active .count { background: var(--accent-fg); color: var(--accent); }
.tabs { display: flex; gap: var(--s2); margin-bottom: var(--s3); flex-wrap: wrap; }
.msg-list { display: flex; flex-direction: column; gap: var(--s2); }
.msg-row { display: block; text-decoration: none; color: inherit; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s3) var(--s4); box-shadow: var(--shadow); }
.msg-row:hover { border-color: var(--border-strong); }
.msg-row.unread { border-left: 4px solid var(--focus); background: var(--info-bg); }
.msg-row.unread .subj { font-weight: 700; }
.msg-row .top { display: flex; justify-content: space-between; gap: var(--s3); align-items: baseline; flex-wrap: wrap; }
.msg-row .subj { font-weight: 600; }
.msg-row .meta { font-size: var(--fs-sm); color: var(--muted); margin-top: var(--s1); }
.msg-row .snippet { font-size: var(--fs-sm); color: var(--muted); margin-top: var(--s2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.thread-msg { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: var(--s4); margin-bottom: var(--s3); box-shadow: var(--shadow); }
.thread-msg .head { display: flex; justify-content: space-between; align-items: flex-start; gap: var(--s3); flex-wrap: wrap; border-bottom: 1px solid var(--border); padding-bottom: var(--s3); margin-bottom: var(--s3); }
.thread-msg .body { white-space: pre-wrap; word-break: break-word; }
.chips { display: flex; gap: var(--s2); flex-wrap: wrap; margin-bottom: var(--s4); }
.chip { font-size: var(--fs-sm); padding: 0.15rem 0.6rem; background: var(--sunken); border: 1px solid var(--border); border-radius: 999px; font-family: var(--mono); }
h1.title { font-size: var(--fs-xl); margin: 0 0 var(--s4); }
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


def _page(*, title: str, prefix: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{_e(title)} — ZOS Inbox</title>
  <style>{_CSS}</style>
</head>
<body>
  <a class="skip" href="#main">Skip to content</a>
  <header class="topbar">
    <div class="topbar-in">
      <div class="row">
        <a href="/admin" class="btn secondary sm">&larr; Admin</a>
        <a class="brand" href="{_e(prefix)}/">Inbox</a>
      </div>
      <a href="{_e(prefix)}/compose" class="btn sm">Compose</a>
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


def create_ui_app(store: InboxStore, *, module_secret: str) -> Starlette:
    """Create the mobile-first server-rendered owner Inbox UI."""

    def _verify_owner(request: Request) -> str | None:
        try:
            caller = verify_http_caller(dict(request.headers), secret=module_secret)
        except CallerVerificationError:
            return None
        return caller.client_name

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "module": "inbox", "service": "ui"})

    async def mailbox_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        csrf_token, is_new = _ensure_csrf(request)

        selected = str(request.path_params.get("mailbox") or request.query_params.get("mailbox") or "owner").strip().lower()
        view_mode = request.query_params.get("view", "all").strip().lower()
        unread_only = view_mode == "unread"
        include_archived = view_mode == "archived"

        mboxes = await store.directory(owner_view=True)
        if selected == "__all__":
            listing = await store.list_all_messages(limit=50)
            messages = listing["messages"]
            if unread_only:
                messages = [m for m in messages if not m["read"]]
            elif view_mode == "archived":
                messages = [m for m in messages if m["archived"]]
        else:
            listing = await store.list_messages(
                caller="owner",
                mailbox=selected,
                unread_only=unread_only,
                include_archived=include_archived,
                limit=50,
                allow_any_mailbox=True,
            )
            messages = listing["messages"]
            if view_mode == "archived":
                messages = [m for m in messages if m["archived"]]

        all_pill = (
            f'<a class="pill {"active" if selected == "__all__" else ""}" '
            f'href="{_e(pref)}/all">All Mail</a>'
        )
        pills_html = all_pill + "".join(
            f'<a class="pill {"active" if mb["mailbox"] == selected else ""}" '
            f'href="{_e(pref)}/mailbox/{quote(mb["mailbox"], safe="")}">'
            f'{_e(mb["mailbox"])}'
            + (f'<span class="count">{int(mb["unread_count"])}</span>' if (mb["unread_count"] or 0) > 0 else "")
            + "</a>"
            for mb in mboxes
        )

        mb_base = f"{pref}/all" if selected == "__all__" else f"{pref}/mailbox/{quote(selected, safe='')}"
        tabs_html = (
            f'<div class="tabs">'
            f'<a class="btn sm {"" if view_mode == "all" else "secondary"}" href="{_e(mb_base)}?view=all">All</a>'
            f'<a class="btn sm {"" if view_mode == "unread" else "secondary"}" href="{_e(mb_base)}?view=unread">Unread</a>'
            f'<a class="btn sm {"" if view_mode == "archived" else "secondary"}" href="{_e(mb_base)}?view=archived">Archived</a>'
            f'</div>'
        )

        notice = request.query_params.get("notice", "").strip()
        notice_html = f'<div class="notice notice-ok">{_e(notice)}</div>' if notice else ""

        rows_html = []
        for m in messages:
            snippet = m["body"].replace("\n", " ").strip()[:120]
            unread_cls = "unread" if not m["read"] else ""
            to_list = ", ".join(m["to"])
            rows_html.append(
                f'<a class="msg-row {unread_cls}" href="{_e(pref)}/thread/{_e(m["thread_id"])}?mb={quote(selected, safe="")}">'
                f'<div class="top"><span class="subj">{_e(m["subject"])}</span>'
                f'<span class="muted">{_e(_fmt_time(m["created_at"]))}</span></div>'
                f'<div class="meta">From <strong>{_e(m["from"])}</strong> &rarr; {_e(to_list)}</div>'
                f'<div class="snippet">{_e(snippet)}</div>'
                f'</a>'
            )

        list_html = (
            f'<div class="msg-list">{"".join(rows_html)}</div>'
            if rows_html
            else f'<div class="card"><p class="empty">No messages in <code>{_e(selected)}</code>.</p></div>'
        )

        body = f"""
        {notice_html}
        <div class="pills">{pills_html}</div>
        {tabs_html}
        {list_html}
        """
        resp = HTMLResponse(_page(title=f"Mailbox: {selected}", prefix=pref, body=body))
        return _attach_csrf_cookie(resp, csrf_token, prefix=pref, is_new=is_new)

    async def thread_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        csrf_token, is_new = _ensure_csrf(request)
        thread_id = str(request.path_params.get("thread_id", "")).strip()
        back_mb = request.query_params.get("mb", "owner").strip().lower() or "owner"

        try:
            msgs = await store.get_thread(caller="owner", thread_id=thread_id, mark_read=True, allow_any=True)
        except ValueError:
            return HTMLResponse(_page(title="Not Found", prefix=pref, body='<div class="card"><p>Invalid thread ID.</p></div>'), status_code=404)

        if not msgs:
            return HTMLResponse(_page(title="Not Found", prefix=pref, body='<div class="card"><p>Thread not found.</p></div>'), status_code=404)

        subject = msgs[0]["subject"]
        reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        last_msg = msgs[-1]
        reply_targets = sorted({r for r in [last_msg["from"], *last_msg["to"]] if r != "owner"} or {"general"})

        cards = []
        for m in msgs:
            arch_btn = ""
            if not m["archived"]:
                arch_btn = (
                    f'<form method="post" action="{_e(pref)}/archive">'
                    f'<input type="hidden" name="csrf_token" value="{_e(csrf_token)}">'
                    f'<input type="hidden" name="id" value="{int(m["id"])}">'
                    f'<input type="hidden" name="mb" value="{_e(back_mb)}">'
                    f'<button type="submit" class="secondary sm">Archive</button>'
                    f'</form>'
                )
            cards.append(
                f'<div class="thread-msg">'
                f'<div class="head">'
                f'<div><strong>{_e(m["from"])}</strong> &rarr; <code>{_e(", ".join(m["to"]))}</code>'
                f'<div class="muted">{_e(_fmt_time(m["created_at"]))}</div></div>'
                f'{arch_btn}'
                f'</div>'
                f'<div class="body">{_e(m["body"])}</div>'
                f'</div>'
            )

        back_href = f"{pref}/all" if back_mb == "__all__" else f"{pref}/mailbox/{quote(back_mb, safe='')}"
        back_label = "All Mail" if back_mb == "__all__" else back_mb
        body = f"""
        <p class="back"><a class="btn secondary sm" href="{_e(back_href)}">&larr; {_e(back_label)}</a></p>
        <h1 class="title">{_e(subject)}</h1>
        {"".join(cards)}
        <div class="card">
          <h2>Reply as <code>owner</code></h2>
          <form method="post" action="{_e(pref)}/send">
            <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
            <input type="hidden" name="thread_id" value="{_e(thread_id)}">
            <label class="field block">To (comma-separated mailboxes)
              <input type="text" name="to" value="{_e(', '.join(reply_targets))}" required>
            </label>
            <label class="field block">Subject
              <input type="text" name="subject" value="{_e(reply_subject)}" required>
            </label>
            <label class="field block">Message
              <textarea name="body" required placeholder="Write your reply..."></textarea>
            </label>
            <button type="submit" class="block">Send Reply</button>
          </form>
        </div>
        """
        resp = HTMLResponse(_page(title=subject, prefix=pref, body=body))
        return _attach_csrf_cookie(resp, csrf_token, prefix=pref, is_new=is_new)

    async def compose_view(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        csrf_token, is_new = _ensure_csrf(request)
        mboxes = await store.directory(owner_view=True)
        chips = "".join(f'<span class="chip">{_e(m["mailbox"])}</span>' for m in mboxes)
        err = request.query_params.get("error", "").strip()
        err_html = f'<div class="notice notice-err">{_e(err)}</div>' if err else ""
        prefill_to = request.query_params.get("to", "").strip()

        body = f"""
        <p class="back"><a class="btn secondary sm" href="{_e(pref)}/">&larr; Back to Inbox</a></p>
        <div class="card">
          <h1>New Message (as <code>owner</code>)</h1>
          {err_html}
          <div class="muted block">Known mailboxes</div>
          <div class="chips">{chips}</div>
          <form method="post" action="{_e(pref)}/send">
            <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
            <label class="field block">To (comma-separated mailboxes)
              <input type="text" name="to" value="{_e(prefill_to)}" placeholder="general, claude, codex" required autofocus>
            </label>
            <label class="field block">Subject
              <input type="text" name="subject" placeholder="Subject" required>
            </label>
            <label class="field block">Body
              <textarea name="body" placeholder="Write your message..." required></textarea>
            </label>
            <button type="submit" class="block">Send Message</button>
          </form>
        </div>
        """
        resp = HTMLResponse(_page(title="Compose", prefix=pref, body=body))
        return _attach_csrf_cookie(resp, csrf_token, prefix=pref, is_new=is_new)

    async def send_action(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        form = await request.form()
        cookie_csrf = request.cookies.get(CSRF_COOKIE, "").strip()
        form_csrf = str(form.get("csrf_token", "")).strip()
        if not cookie_csrf or not form_csrf or not hmac.compare_digest(cookie_csrf, form_csrf):
            return HTMLResponse("<h1>Invalid CSRF token</h1>", status_code=403)

        raw_to = str(form.get("to", ""))
        recipients = [part.strip() for part in raw_to.split(",") if part.strip()]
        subject = str(form.get("subject", "")).strip()
        body_text = str(form.get("body", ""))
        thread_id = str(form.get("thread_id", "")).strip() or None

        try:
            msg = await store.send_message(
                sender="owner",
                to=recipients,
                subject=subject,
                body=body_text,
                thread_id=thread_id,
            )
        except Exception as exc:
            return RedirectResponse(f"{pref}/compose?error={quote(str(exc))}", status_code=303)

        return RedirectResponse(f"{pref}/thread/{msg['thread_id']}", status_code=303)

    async def archive_action(request: Request) -> Response:
        if _verify_owner(request) is None:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        pref = _prefix(request)
        form = await request.form()
        cookie_csrf = request.cookies.get(CSRF_COOKIE, "").strip()
        form_csrf = str(form.get("csrf_token", "")).strip()
        if not cookie_csrf or not form_csrf or not hmac.compare_digest(cookie_csrf, form_csrf):
            return HTMLResponse("<h1>Invalid CSRF token</h1>", status_code=403)

        mb = str(form.get("mb", "owner")).strip().lower() or "owner"
        with contextlib.suppress(Exception):
            msg_id = int(str(form.get("id", "")))
            await store.archive_message(caller="owner", message_id=msg_id, allow_any=True)
        return RedirectResponse(
            f"{pref}/mailbox/{quote(mb, safe='')}?notice={quote('Message archived.')}",
            status_code=303,
        )

    async def all_mail_view(request: Request) -> Response:
        request.path_params["mailbox"] = "__all__"
        return await mailbox_view(request)

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/", mailbox_view, methods=["GET"]),
            Route("/all", all_mail_view, methods=["GET"]),
            Route("/mailbox/{mailbox:str}", mailbox_view, methods=["GET"]),
            Route("/thread/{thread_id:str}", thread_view, methods=["GET"]),
            Route("/compose", compose_view, methods=["GET"]),
            Route("/send", send_action, methods=["POST"]),
            Route("/archive", archive_action, methods=["POST"]),
        ]
    )
