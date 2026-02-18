from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, now_utc
from .models import Entry, EntryStatus, Insight, Job, JobStatus, Transcript, User

router = APIRouter(prefix="/admin", tags=["admin"])
basic_security = HTTPBasic()


ADMIN_AUTH_HEADER = {"WWW-Authenticate": 'Basic realm="theVoid Admin"'}


def require_admin(credentials: HTTPBasicCredentials = Depends(basic_security)) -> None:
    if not settings.admin_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin viewer disabled")

    username_ok = secrets.compare_digest(credentials.username, settings.admin_username)
    password_ok = secrets.compare_digest(credentials.password, settings.admin_password)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers=ADMIN_AUTH_HEADER,
        )


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _snippet(text: str | None, max_chars: int = 180) -> str:
    if not text:
        return "-"
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _layout(title: str, body: str) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(title)} - theVoid Admin</title>
  <style>
    :root {{
      --bg: #11141a;
      --card: #191f2a;
      --line: #2a3342;
      --text: #edf1f7;
      --muted: #97a3b6;
      --accent: #18a999;
      --danger: #b34859;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 14px 20px; border-bottom: 1px solid var(--line); display: flex; gap: 14px; align-items: center; }}
    header strong {{ font-size: 15px; }}
    header a {{ color: var(--muted); text-decoration: none; font-size: 14px; }}
    header a:hover {{ color: var(--text); }}
    main {{ padding: 18px 20px 26px; max-width: 1320px; margin: 0 auto; }}
    h1 {{ margin: 0 0 14px; font-size: 22px; }}
    .meta {{ color: var(--muted); margin-bottom: 16px; font-size: 13px; }}
    .cards {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 14px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .value {{ font-size: 24px; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .chip {{ display: inline-block; border-radius: 999px; border: 1px solid var(--line); padding: 3px 8px; font-size: 11px; color: var(--muted); }}
    .chip.ready {{ color: var(--accent); border-color: var(--accent); }}
    .chip.failed {{ color: var(--danger); border-color: var(--danger); }}
    a.link {{ color: var(--accent); text-decoration: none; }}
    a.link:hover {{ text-decoration: underline; }}
    form {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
    input, button {{ background: #121722; border: 1px solid var(--line); color: var(--text); border-radius: 8px; padding: 8px 10px; font-size: 13px; }}
    button {{ background: #1f2938; cursor: pointer; }}
    pre {{ white-space: pre-wrap; line-height: 1.35; background: #0e131d; border: 1px solid var(--line); padding: 12px; border-radius: 10px; }}
    .empty {{ color: var(--muted); margin: 12px 0; }}
  </style>
</head>
<body>
  <header>
    <strong>theVoid Admin</strong>
    <a href=\"/admin\">Overview</a>
    <a href=\"/admin/transcripts\">Transcripts</a>
  </header>
  <main>
    {body}
  </main>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("", response_class=HTMLResponse)
def admin_overview(
    limit: int = Query(25, ge=1, le=150),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    users_count = db.query(func.count(User.id)).scalar() or 0
    entries_count = db.query(func.count(Entry.id)).scalar() or 0
    ready_entries_count = (
        db.query(func.count(Entry.id)).filter(Entry.status == EntryStatus.READY).scalar() or 0
    )
    transcript_count = db.query(func.count(Transcript.entry_id)).scalar() or 0
    failed_jobs_24h = (
        db.query(func.count(Job.id))
        .filter(and_(Job.status == JobStatus.FAILED, Job.updated_at >= now_utc() - timedelta(hours=24)))
        .scalar()
        or 0
    )

    rows = (
        db.query(Entry, User, Transcript, Insight)
        .join(User, User.id == Entry.user_id)
        .outerjoin(Transcript, Transcript.entry_id == Entry.id)
        .outerjoin(Insight, Insight.entry_id == Entry.id)
        .order_by(Entry.created_at.desc())
        .limit(limit)
        .all()
    )

    table_rows: list[str] = []
    for entry, user, transcript, insight in rows:
        status_class = "ready" if entry.status == EntryStatus.READY else "failed" if entry.status == EntryStatus.FAILED else ""
        mood = f"{insight.mood_score:.1f}" if insight is not None else "-"
        table_rows.append(
            "<tr>"
            f"<td><a class='link' href='/admin/entries/{escape(entry.id)}'>{escape(entry.id[:8])}</a></td>"
            f"<td>{escape(user.anonymous_handle)}</td>"
            f"<td>{escape(str(entry.local_date))}</td>"
            f"<td><span class='chip {status_class}'>{escape(entry.status.value)}</span></td>"
            f"<td>{escape(mood)}</td>"
            f"<td>{escape(_snippet(transcript.text if transcript else None, 120))}</td>"
            f"<td>{escape(_fmt_dt(entry.created_at))}</td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr>"
        "<th>Entry</th><th>User</th><th>Local Date</th><th>Status</th><th>Mood</th><th>Transcript Snippet</th><th>Created</th>"
        "</tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        if table_rows
        else "<p class='empty'>No entries yet.</p>"
    )

    body = f"""
      <h1>Overview</h1>
      <p class=\"meta\">Now: {escape(_fmt_dt(now_utc()))}</p>
      <section class=\"cards\">
        <div class=\"card\"><div class=\"label\">Users</div><div class=\"value\">{users_count}</div></div>
        <div class=\"card\"><div class=\"label\">Entries</div><div class=\"value\">{entries_count}</div></div>
        <div class=\"card\"><div class=\"label\">Ready Entries</div><div class=\"value\">{ready_entries_count}</div></div>
        <div class=\"card\"><div class=\"label\">Transcripts</div><div class=\"value\">{transcript_count}</div></div>
        <div class=\"card\"><div class=\"label\">Failed Jobs (24h)</div><div class=\"value\">{failed_jobs_24h}</div></div>
      </section>
      <h1 style=\"margin-top: 18px;\">Latest Entries</h1>
      {table_html}
    """

    return _layout("Overview", body)


@router.get("/transcripts", response_class=HTMLResponse)
def admin_transcripts(
    q: str = Query(default="", max_length=200),
    local_date: date | None = Query(default=None),
    limit: int = Query(75, ge=1, le=300),
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    query = (
        db.query(Entry, User, Transcript, Insight)
        .join(User, User.id == Entry.user_id)
        .join(Transcript, Transcript.entry_id == Entry.id)
        .outerjoin(Insight, Insight.entry_id == Entry.id)
    )

    if q.strip():
        query = query.filter(Transcript.text.ilike(f"%{q.strip()}%"))

    if local_date is not None:
        query = query.filter(Entry.local_date == local_date)

    rows = query.order_by(Entry.created_at.desc()).limit(limit).all()

    row_html: list[str] = []
    for entry, user, transcript, insight in rows:
        mood = f"{insight.mood_score:.1f}" if insight is not None else "-"
        row_html.append(
            "<tr>"
            f"<td><a class='link' href='/admin/entries/{escape(entry.id)}'>{escape(entry.id[:8])}</a></td>"
            f"<td>{escape(user.anonymous_handle)}</td>"
            f"<td>{escape(str(entry.local_date))}</td>"
            f"<td>{escape(mood)}</td>"
            f"<td>{escape(_snippet(transcript.text, 220))}</td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr><th>Entry</th><th>User</th><th>Local Date</th><th>Mood</th><th>Transcript</th></tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        if row_html
        else "<p class='empty'>No transcript rows matched your filters.</p>"
    )

    local_date_value = local_date.isoformat() if local_date else ""

    body = f"""
      <h1>Transcripts</h1>
      <form method=\"get\">
        <input type=\"text\" name=\"q\" value=\"{escape(q)}\" placeholder=\"Search transcript text\" />
        <input type=\"date\" name=\"local_date\" value=\"{escape(local_date_value)}\" />
        <input type=\"number\" name=\"limit\" min=\"1\" max=\"300\" value=\"{limit}\" />
        <button type=\"submit\">Apply</button>
      </form>
      {table_html}
    """

    return _layout("Transcripts", body)


@router.get("/entries/{entry_id}", response_class=HTMLResponse)
def admin_entry_detail(
    entry_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    row = (
        db.query(Entry, User, Transcript, Insight)
        .join(User, User.id == Entry.user_id)
        .outerjoin(Transcript, Transcript.entry_id == Entry.id)
        .outerjoin(Insight, Insight.entry_id == Entry.id)
        .filter(Entry.id == entry_id)
        .one_or_none()
    )

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")

    entry, user, transcript, insight = row
    transcript_text = transcript.text if transcript is not None else "No transcript yet."
    mood = f"{insight.mood_score:.1f}" if insight is not None else "-"
    tags = ", ".join(insight.mood_tags) if insight is not None and insight.mood_tags else "-"

    body = f"""
      <h1>Entry {escape(entry.id)}</h1>
      <p class=\"meta\">User: @{escape(user.anonymous_handle)} | Created: {escape(_fmt_dt(entry.created_at))}</p>
      <section class=\"cards\">
        <div class=\"card\"><div class=\"label\">Status</div><div class=\"value\" style=\"font-size: 18px;\">{escape(entry.status.value)}</div></div>
        <div class=\"card\"><div class=\"label\">Local Date</div><div class=\"value\" style=\"font-size: 18px;\">{escape(str(entry.local_date))}</div></div>
        <div class=\"card\"><div class=\"label\">Duration (s)</div><div class=\"value\" style=\"font-size: 18px;\">{entry.duration_seconds}</div></div>
        <div class=\"card\"><div class=\"label\">Mood</div><div class=\"value\" style=\"font-size: 18px;\">{escape(mood)}</div></div>
      </section>
      <p class=\"meta\">Tags: {escape(tags)} | Audio Object Key: {escape(entry.audio_object_key)}</p>
      <h1 style=\"margin-top: 10px; font-size: 18px;\">Transcript</h1>
      <pre>{escape(transcript_text)}</pre>
    """

    return _layout(f"Entry {entry.id}", body)
