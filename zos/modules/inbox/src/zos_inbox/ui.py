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

_CSS = """
:root { --bg:#fafafa; --fg:#111827; --muted:#6b7280; --border:#e5e7eb; --card:#ffffff;
        --accent:#111827; --unread:#2563eb; --unread-bg:#eff6ff; --ok-bg:#dcfce7; --ok:#15803d;
        --err-bg:#fee2e2; --err:#b91c1c; --mono:ui-monospace,SFMono-Regular,Menlo,monospace; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.45; }
header { background: var(--card); border-bottom: 1px solid var(--border); padding: 0.7rem 1rem; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap; position: sticky; top: 0; z-index: 10; }
header .title { font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem; }
header .actions { display: flex; gap: 0.5rem; align-items: center; }
main { max-width: 920px; margin: 1rem auto; padding: 0 0.85rem 3rem; }
.pills { display: flex; gap: 0.45rem; overflow-x: auto; padding-bottom: 0.4rem; margin-bottom: 0.9rem; -webkit-overflow-scrolling: touch; }
.pill { display: inline-flex; align-items: center; gap: 0.35rem; white-space: nowrap; text-decoration: none; color: var(--fg); background: var(--card); border: 1px solid var(--border); padding: 0.4rem 0.75rem; border-radius: 999px; font-size: 0.88rem; }
.pill.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.pill .count { background: var(--unread); color: #fff; border-radius: 999px; padding: 0.05rem 0.45rem; font-size: 0.75rem; font-weight: 700; }
.pill.active .count { background: #fff; color: var(--accent); }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; margin-bottom: 0.9rem; }
.msg-list { display: flex; flex-direction: column; gap: 0.55rem; }
.msg-row { display: block; text-decoration: none; color: inherit; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 0.85rem 1rem; }
.msg-row.unread { border-left: 4px solid var(--unread); background: var(--unread-bg); }
.msg-row .top { display: flex; justify-content: space-between; gap: 0.5rem; align-items: baseline; }
.msg-row .subj { font-weight: 600; font-size: 0.96rem; }
.msg-row .meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }
.msg-row .snippet { font-size: 0.86rem; color: #374151; margin-top: 0.35rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.thread-msg { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem; }
.thread-msg .head { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem; flex-wrap: wrap; border-bottom: 1px solid var(--border); padding-bottom: 0.55rem; margin-bottom: 0.7rem; }
.thread-msg .body { white-space: pre-wrap; word-break: break-word; font-size: 0.95rem; }
.btn, button { display: inline-block; text-decoration: none; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--accent); background: var(--accent); color: #fff; font-size: 0.88rem; font-weight: 600; cursor: pointer; }
.btn.secondary, button.secondary { background: #fff; color: var(--fg); border-color: #d1d5db; }
.btn.sm, button.sm { padding: 0.3rem 0.6rem; font-size: 0.78rem; }
label.field { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.86rem; font-weight: 600; margin-bottom: 0.85rem; }
input[type=text], textarea { width: 100%; padding: 0.6rem 0.7rem; border: 1px solid #d1d5db; border-radius: 7px; font-size: 0.95rem; font-family: inherit; background: #fff; }
textarea { min-height: 130px; resize: vertical; }
.notice { padding: 0.7rem 0.9rem; border-radius: 8px; margin-bottom: 0.9rem; font-size: 0.9rem; border: 1px solid; }
.notice-ok { background: var(--ok-bg); color: var(--ok); border-color: #86efac; }
.notice-err { background: var(--err-bg); color: var(--err); border-color: #fca5a5; }
.muted { color: var(--muted); font-size: 0.85rem; }
.chips { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
.chip { font-size: 0.8rem; padding: 0.25rem 0.55rem; background: #f3f4f6; border: 1px solid var(--border); border-radius: 999px; font-family: var(--mono); }
.tabs { display: flex; gap: 0.4rem; margin-bottom: 0.75rem; }
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
  <title>{_e(title)} — ZOS Inbox</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <div class="title">
      <a href="/admin" class="btn secondary sm">&larr; Admin</a>
      <a href="{_e(prefix)}/" style="text-decoration:none;color:inherit;">Inbox</a>
    </div>
    <div class="actions">
      <a href="{_e(prefix)}/compose" class="btn sm">Compose</a>
    </div>
  </header>
  <main>{body}</main>
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
            else f'<div class="card"><p class="muted" style="margin:0;">No messages in <code>{_e(selected)}</code>.</p></div>'
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
                    f'<form method="post" action="{_e(pref)}/archive" style="margin:0;">'
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
        <p style="margin:0 0 0.7rem;"><a class="btn secondary sm" href="{_e(back_href)}">&larr; {_e(back_label)}</a></p>
        <h1 style="font-size:1.25rem;margin:0 0 0.85rem;">{_e(subject)}</h1>
        {"".join(cards)}
        <div class="card">
          <h2 style="font-size:1rem;margin:0 0 0.7rem;">Reply as <code>owner</code></h2>
          <form method="post" action="{_e(pref)}/send">
            <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
            <input type="hidden" name="thread_id" value="{_e(thread_id)}">
            <label class="field">To (comma-separated mailboxes)
              <input type="text" name="to" value="{_e(', '.join(reply_targets))}" required>
            </label>
            <label class="field">Subject
              <input type="text" name="subject" value="{_e(reply_subject)}" required>
            </label>
            <label class="field">Message
              <textarea name="body" required placeholder="Write your reply..."></textarea>
            </label>
            <button type="submit">Send Reply</button>
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
        <p style="margin:0 0 0.7rem;"><a class="btn secondary sm" href="{_e(pref)}/">&larr; Back to Inbox</a></p>
        <div class="card">
          <h1 style="font-size:1.2rem;margin:0 0 0.65rem;">New Message (as <code>owner</code>)</h1>
          {err_html}
          <div class="muted" style="margin-bottom:0.35rem;">Known mailboxes:</div>
          <div class="chips">{chips}</div>
          <form method="post" action="{_e(pref)}/send">
            <input type="hidden" name="csrf_token" value="{_e(csrf_token)}">
            <label class="field">To (comma-separated mailboxes)
              <input type="text" name="to" value="{_e(prefill_to)}" placeholder="general, claude, codex" required autofocus>
            </label>
            <label class="field">Subject
              <input type="text" name="subject" placeholder="Subject" required>
            </label>
            <label class="field">Body
              <textarea name="body" placeholder="Write your message..." required></textarea>
            </label>
            <button type="submit">Send Message</button>
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
