from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
import json
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from .config import settings
from .db import get_admin_db, now_utc
from .models import (
    AccountDecommission,
    Entry,
    EntryStatus,
    FeedbackReport,
    FriendEdge,
    Insight,
    Job,
    JobStatus,
    SocialDotEvent,
    Transcript,
    User,
)
from .social import SILENT_DOT_COLOR

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


def _safe_redirect_target(raw: str | None, fallback: str = "/admin/users") -> str:
    if raw is None:
        return fallback
    target = raw.strip()
    if not target.startswith("/admin"):
        return fallback
    return target or fallback


def _safe_dot_color(value: str | None) -> str:
    if value is None:
        return SILENT_DOT_COLOR

    normalized = value.strip()
    if (
        len(normalized) == 7
        and normalized.startswith("#")
        and all(char in "0123456789abcdefABCDEF" for char in normalized[1:])
    ):
        return normalized.upper()

    return SILENT_DOT_COLOR


def _display_name_or_handle(user: User) -> str:
    display_name = (user.display_name or "").strip()
    if display_name:
        return display_name

    handle = (user.anonymous_handle or "").strip()
    if handle:
        return f"@{handle}"

    return user.id[:8]


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
    .chip.user-active {{ color: var(--accent); border-color: var(--accent); }}
    .chip.user-decommissioned {{ color: var(--danger); border-color: var(--danger); }}
    .dot-swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 999px; border: 1px solid rgba(255,255,255,.28); margin-right: 8px; vertical-align: -2px; }}
    a.link {{ color: var(--accent); text-decoration: none; }}
    a.link:hover {{ text-decoration: underline; }}
    form {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
    input, button, select {{ background: #121722; border: 1px solid var(--line); color: var(--text); border-radius: 8px; padding: 8px 10px; font-size: 13px; }}
    button {{ background: #1f2938; cursor: pointer; }}
    button.danger {{ border-color: #6f3340; color: #f6d7dc; }}
    button.ok {{ border-color: #257062; color: #d8fff8; }}
    .inline-form {{ display: inline-flex; margin: 0; gap: 6px; align-items: center; }}
    pre {{ white-space: pre-wrap; line-height: 1.35; background: #0e131d; border: 1px solid var(--line); padding: 12px; border-radius: 10px; }}
    .empty {{ color: var(--muted); margin: 12px 0; }}
    .graph-canvas {{ width: 100%; min-height: 420px; border: 1px solid var(--line); border-radius: 10px; background: #0e131d; padding: 6px; }}
    .graph-canvas svg {{ width: 100%; height: 420px; display: block; }}
  </style>
</head>
<body>
  <header>
    <strong>theVoid Admin</strong>
    <a href=\"/admin\">Overview</a>
    <a href=\"/admin/dots\">Dots</a>
    <a href=\"/admin/users\">Users</a>
    <a href=\"/admin/feedback\">Feedback</a>
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
    db: Session = Depends(get_admin_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    users_count = db.query(func.count(User.id)).scalar() or 0
    decommissioned_users_count = db.query(func.count(AccountDecommission.user_id)).scalar() or 0
    active_users_count = max(0, users_count - decommissioned_users_count)
    entries_count = db.query(func.count(Entry.id)).scalar() or 0
    ready_entries_count = (
        db.query(func.count(Entry.id)).filter(Entry.status == EntryStatus.READY).scalar() or 0
    )
    failed_jobs_24h = (
        db.query(func.count(Job.id))
        .filter(and_(Job.status == JobStatus.FAILED, Job.updated_at >= now_utc() - timedelta(hours=24)))
        .scalar()
        or 0
    )
    feedback_count = db.query(func.count(FeedbackReport.id)).scalar() or 0
    feedback_24h = (
        db.query(func.count(FeedbackReport.id))
        .filter(FeedbackReport.created_at >= now_utc() - timedelta(hours=24))
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

    decommissioned_rows = (
        db.query(AccountDecommission, User)
        .join(User, User.id == AccountDecommission.user_id)
        .order_by(AccountDecommission.decommissioned_at.desc())
        .limit(30)
        .all()
    )

    decommissioned_html_rows: list[str] = []
    for decommission, user in decommissioned_rows:
        reason = _snippet(decommission.reason, 120)
        decommissioned_html_rows.append(
            "<tr>"
            f"<td>{escape(user.id[:8])}</td>"
            f"<td>@{escape(user.anonymous_handle)}</td>"
            f"<td>{escape(user.display_name or '-')}</td>"
            f"<td>{escape(reason)}</td>"
            f"<td>{escape(_fmt_dt(decommission.decommissioned_at))}</td>"
            f"<td>"
            f"<form class='inline-form' method='post' action='/admin/users/{escape(user.id)}/recommission'>"
            "<input type='hidden' name='redirect_to' value='/admin' />"
            "<button class='ok' type='submit'>Recommission</button>"
            "</form>"
            "</td>"
            "</tr>"
        )

    decommissioned_html = (
        "<table>"
        "<thead><tr><th>User</th><th>Handle</th><th>Display Name</th><th>Reason</th><th>Decommissioned At</th><th>Action</th></tr></thead>"
        f"<tbody>{''.join(decommissioned_html_rows)}</tbody>"
        "</table>"
        if decommissioned_html_rows
        else "<p class='empty'>No decommissioned users.</p>"
    )

    feedback_rows = (
        db.query(FeedbackReport, User)
        .join(User, User.id == FeedbackReport.user_id)
        .order_by(FeedbackReport.created_at.desc(), FeedbackReport.id.desc())
        .limit(20)
        .all()
    )

    feedback_html_rows: list[str] = []
    for report, user in feedback_rows:
        feedback_html_rows.append(
            "<tr>"
            f"<td>{escape(user.id[:8])}</td>"
            f"<td>@{escape(user.anonymous_handle)}</td>"
            f"<td><span class='chip'>{escape(report.kind)}</span></td>"
            f"<td>{escape(_snippet(report.message, 160))}</td>"
            f"<td>{escape(_fmt_dt(report.created_at))}</td>"
            "</tr>"
        )

    feedback_html = (
        "<table>"
        "<thead><tr><th>User</th><th>Handle</th><th>Type</th><th>Message</th><th>Created</th></tr></thead>"
        f"<tbody>{''.join(feedback_html_rows)}</tbody>"
        "</table>"
        if feedback_html_rows
        else "<p class='empty'>No feedback reports yet.</p>"
    )

    body = f"""
      <h1>Overview</h1>
      <p class=\"meta\">Now: {escape(_fmt_dt(now_utc()))}</p>
      <section class=\"cards\">
        <div class=\"card\"><div class=\"label\">Users</div><div class=\"value\">{users_count}</div></div>
        <div class=\"card\"><div class=\"label\">Active Users</div><div class=\"value\">{active_users_count}</div></div>
        <div class=\"card\"><div class=\"label\">Decommissioned</div><div class=\"value\">{decommissioned_users_count}</div></div>
        <div class=\"card\"><div class=\"label\">Entries</div><div class=\"value\">{entries_count}</div></div>
        <div class=\"card\"><div class=\"label\">Ready Entries</div><div class=\"value\">{ready_entries_count}</div></div>
        <div class=\"card\"><div class=\"label\">Failed Jobs (24h)</div><div class=\"value\">{failed_jobs_24h}</div></div>
        <div class=\"card\"><div class=\"label\">Feedback Reports</div><div class=\"value\">{feedback_count}</div></div>
        <div class=\"card\"><div class=\"label\">Feedback (24h)</div><div class=\"value\">{feedback_24h}</div></div>
      </section>
      <h1 style=\"margin-top: 18px;\">Latest Entries</h1>
      {table_html}
      <h1 style=\"margin-top: 18px;\">Decommissioned Accounts</h1>
      {decommissioned_html}
      <h1 style=\"margin-top: 18px;\">Latest Feedback</h1>
      {feedback_html}
      <p class=\"meta\"><a class='link' href='/admin/feedback'>Open full feedback view</a></p>
    """

    return _layout("Overview", body)


@router.get("/dots", response_class=HTMLResponse)
def admin_dots(
    q: str = Query(default="", max_length=200),
    local_date: date | None = Query(default=None),
    latest_only: bool = Query(default=False),
    limit: int = Query(150, ge=1, le=500),
    db: Session = Depends(get_admin_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    query = db.query(SocialDotEvent, User).join(User, User.id == SocialDotEvent.user_id)

    if local_date is not None:
        query = query.filter(SocialDotEvent.local_date == local_date)

    if q.strip():
        pattern = f"%{q.strip()}%"
        query = query.filter(
            or_(
                SocialDotEvent.user_id.ilike(pattern),
                User.anonymous_handle.ilike(pattern),
                User.display_name.ilike(pattern),
                SocialDotEvent.dot_color.ilike(pattern),
            )
        )

    total_matching = query.count()

    rows: list[tuple[SocialDotEvent, User]]
    if latest_only:
        ordered_rows = (
            query.order_by(
                SocialDotEvent.user_id.asc(),
                SocialDotEvent.updated_at.desc(),
                SocialDotEvent.local_date.desc(),
                SocialDotEvent.id.desc(),
            )
            .limit(max(1000, limit * 4))
            .all()
        )
        latest_by_user: dict[str, tuple[SocialDotEvent, User]] = {}
        for dot_event, user in ordered_rows:
            if dot_event.user_id not in latest_by_user:
                latest_by_user[dot_event.user_id] = (dot_event, user)

        rows = sorted(
            latest_by_user.values(),
            key=lambda item: (item[0].updated_at, item[0].local_date, item[0].user_id, item[0].id),
            reverse=True,
        )[:limit]
    else:
        rows = (
            query.order_by(
                SocialDotEvent.updated_at.desc(),
                SocialDotEvent.local_date.desc(),
                SocialDotEvent.user_id.asc(),
                SocialDotEvent.id.desc(),
            )
            .limit(limit)
            .all()
        )

    unique_users = len({dot_event.user_id for dot_event, _ in rows})
    muted_count = sum(1 for dot_event, _ in rows if _safe_dot_color(dot_event.dot_color) == SILENT_DOT_COLOR)

    row_html: list[str] = []
    for dot_event, user in rows:
        color = _safe_dot_color(dot_event.dot_color)
        color_html = (
            f"<span class='dot-swatch' style='background:{escape(color)};'></span>"
            f"{escape(color)}"
        )
        tags = ", ".join(dot_event.dot_tags) if dot_event.dot_tags else "-"
        display_name = user.display_name or "-"
        row_html.append(
            "<tr>"
            f"<td>{escape(user.id[:8])}</td>"
            f"<td>@{escape(user.anonymous_handle)}</td>"
            f"<td>{escape(display_name)}</td>"
            f"<td>{escape(str(dot_event.local_date))}</td>"
            f"<td>{color_html}</td>"
            f"<td>{escape(tags)}</td>"
            f"<td>{escape(_fmt_dt(dot_event.updated_at))}</td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr><th>User</th><th>Handle</th><th>Display Name</th><th>Local Date</th><th>Dot Color</th><th>Tags</th><th>Updated</th></tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        if row_html
        else "<p class='empty'>No social dots matched your filters.</p>"
    )

    directed_edges = {
        (user_id, friend_user_id)
        for user_id, friend_user_id in db.query(FriendEdge.user_id, FriendEdge.friend_user_id).all()
        if user_id != friend_user_id
    }
    undirected_edges = sorted(
        (user_id, friend_user_id)
        for user_id, friend_user_id in directed_edges
        if user_id < friend_user_id and (friend_user_id, user_id) in directed_edges
    )
    connected_user_ids = sorted({user_id for pair in undirected_edges for user_id in pair})
    graph_users = db.query(User).filter(User.id.in_(connected_user_ids)).all() if connected_user_ids else []
    graph_user_by_id = {user.id: user for user in graph_users}

    degree_by_user: dict[str, int] = {user_id: 0 for user_id in connected_user_ids}
    for left_user_id, right_user_id in undirected_edges:
        degree_by_user[left_user_id] = degree_by_user.get(left_user_id, 0) + 1
        degree_by_user[right_user_id] = degree_by_user.get(right_user_id, 0) + 1

    graph_nodes_payload = []
    for user_id in connected_user_ids:
        user = graph_user_by_id.get(user_id)
        label = _display_name_or_handle(user) if user is not None else user_id[:8]
        graph_nodes_payload.append(
            {
                "id": user_id,
                "label": label,
                "degree": degree_by_user.get(user_id, 0),
            }
        )

    graph_edges_payload = [
        {"source": left_user_id, "target": right_user_id}
        for left_user_id, right_user_id in undirected_edges
    ]

    graph_nodes_json = json.dumps(graph_nodes_payload, ensure_ascii=True).replace("</", "<\\/")
    graph_edges_json = json.dumps(graph_edges_payload, ensure_ascii=True).replace("</", "<\\/")

    graph_script = ""
    if graph_nodes_payload:
        graph_script = """
      <script>
      (function () {
        const nodes = %s;
        const edges = %s;
        const container = document.getElementById("friend-graph");
        if (!container) {
          return;
        }

        const escapeXml = (value) => String(value).replace(/[&<>"']/g, (char) => {
          if (char === "&") return "&amp;";
          if (char === "<") return "&lt;";
          if (char === ">") return "&gt;";
          if (char === '"') return "&quot;";
          return "&#39;";
        });

        const render = () => {
          const width = Math.max(320, container.clientWidth - 12);
          const height = 420;
          const centerX = width / 2;
          const centerY = height / 2;
          const radius = Math.max(90, Math.min(width, height) / 2 - 56);
          const sortedNodes = [...nodes].sort((left, right) => {
            const degreeDelta = right.degree - left.degree;
            if (degreeDelta !== 0) return degreeDelta;
            return left.label.localeCompare(right.label);
          });

          sortedNodes.forEach((node, index) => {
            const angle = ((index / sortedNodes.length) * Math.PI * 2) - (Math.PI / 2);
            node.x = centerX + (Math.cos(angle) * radius);
            node.y = centerY + (Math.sin(angle) * radius);
          });

          const nodeById = new Map(sortedNodes.map((node) => [node.id, node]));
          const html = [];
          html.push('<svg viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="Undirected friend graph">');

          for (const edge of edges) {
            const source = nodeById.get(edge.source);
            const target = nodeById.get(edge.target);
            if (!source || !target) continue;
            html.push(
              '<line x1="' + source.x.toFixed(2) + '" y1="' + source.y.toFixed(2) + '" x2="' + target.x.toFixed(2) + '" y2="' + target.y.toFixed(2) + '" stroke="#365067" stroke-width="1.4" opacity="0.78" />'
            );
          }

          for (const node of sortedNodes) {
            const circleRadius = 5 + Math.min(7, node.degree);
            const safeLabel = escapeXml(node.label);
            const safeTitle = safeLabel + ' (friends: ' + node.degree + ')';
            html.push(
              '<circle cx="' + node.x.toFixed(2) + '" cy="' + node.y.toFixed(2) + '" r="' + circleRadius + '" fill="#18a999" stroke="#d7fffa" stroke-width="1.2"><title>' + safeTitle + '</title></circle>'
            );
            html.push(
              '<text x="' + (node.x + circleRadius + 4).toFixed(2) + '" y="' + (node.y + 4).toFixed(2) + '" fill="#d8e1ee" font-size="11">' + safeLabel + '</text>'
            );
          }

          html.push("</svg>");
          container.innerHTML = html.join("");
        };

        let resizeTimer = null;
        window.addEventListener("resize", () => {
          if (resizeTimer) {
            clearTimeout(resizeTimer);
          }
          resizeTimer = window.setTimeout(render, 120);
        });

        render();
      })();
      </script>
        """ % (graph_nodes_json, graph_edges_json)
    graph_placeholder_html = (
        "<p class='empty'>Loading graph...</p>"
        if graph_nodes_payload
        else "<p class='empty'>No reciprocal friendships yet.</p>"
    )

    edge_rows_html: list[str] = []
    for left_user_id, right_user_id in undirected_edges:
        left_user = graph_user_by_id.get(left_user_id)
        right_user = graph_user_by_id.get(right_user_id)
        left_label = _display_name_or_handle(left_user) if left_user is not None else left_user_id[:8]
        right_label = _display_name_or_handle(right_user) if right_user is not None else right_user_id[:8]
        edge_rows_html.append(
            "<tr>"
            f"<td>{escape(left_user_id[:8])}</td>"
            f"<td>{escape(left_label)}</td>"
            f"<td>{escape(right_user_id[:8])}</td>"
            f"<td>{escape(right_label)}</td>"
            "</tr>"
        )

    edge_table_html = (
        "<table>"
        "<thead><tr><th>User A</th><th>Label A</th><th>User B</th><th>Label B</th></tr></thead>"
        f"<tbody>{''.join(edge_rows_html)}</tbody>"
        "</table>"
        if edge_rows_html
        else "<p class='empty'>No reciprocal friendships yet.</p>"
    )

    local_date_value = local_date.isoformat() if local_date else ""
    latest_true_selected = "selected" if latest_only else ""
    latest_false_selected = "selected" if not latest_only else ""

    body = f"""
      <h1>Social Dots</h1>
      <p class=\"meta\">Inspect full dot event stream and friendship topology.</p>
      <section class=\"cards\">
        <div class=\"card\"><div class=\"label\">Showing</div><div class=\"value\">{len(rows)}</div></div>
        <div class=\"card\"><div class=\"label\">Matching Events</div><div class=\"value\">{total_matching}</div></div>
        <div class=\"card\"><div class=\"label\">Users In Result</div><div class=\"value\">{unique_users}</div></div>
        <div class=\"card\"><div class=\"label\">Muted Dots</div><div class=\"value\">{muted_count}</div></div>
        <div class=\"card\"><div class=\"label\">Friend Pairs</div><div class=\"value\">{len(undirected_edges)}</div></div>
        <div class=\"card\"><div class=\"label\">Users In Graph</div><div class=\"value\">{len(connected_user_ids)}</div></div>
      </section>
      <form method=\"get\">
        <input type=\"text\" name=\"q\" value=\"{escape(q)}\" placeholder=\"Search user / handle / color\" />
        <input type=\"date\" name=\"local_date\" value=\"{escape(local_date_value)}\" />
        <select name=\"latest_only\">
          <option value=\"false\" {latest_false_selected}>All events (ordered)</option>
          <option value=\"true\" {latest_true_selected}>Latest per user</option>
        </select>
        <input type=\"number\" name=\"limit\" min=\"1\" max=\"500\" value=\"{limit}\" />
        <button type=\"submit\">Apply</button>
      </form>
      {table_html}
      <h1 style=\"margin-top: 18px;\">Friend Graph</h1>
      <p class=\"meta\">Undirected edges represent reciprocal friendships between users.</p>
      <div id=\"friend-graph\" class=\"graph-canvas\">
        {graph_placeholder_html}
      </div>
      {graph_script}
      <h1 style=\"margin-top: 18px;\">Friend Pairs</h1>
      {edge_table_html}
    """

    return _layout("Dots", body)


@router.get("/users", response_class=HTMLResponse)
def admin_users(
    q: str = Query(default="", max_length=200),
    account_state: str = Query(default="active", pattern="^(active|decommissioned|all)$"),
    limit: int = Query(75, ge=1, le=300),
    db: Session = Depends(get_admin_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    query = (
        db.query(User, AccountDecommission)
        .outerjoin(AccountDecommission, AccountDecommission.user_id == User.id)
    )

    if account_state == "active":
        query = query.filter(AccountDecommission.user_id.is_(None))
    elif account_state == "decommissioned":
        query = query.filter(AccountDecommission.user_id.is_not(None))

    if q.strip():
        pattern = f"%{q.strip()}%"
        query = query.filter(
            or_(
                User.id.ilike(pattern),
                User.apple_sub.ilike(pattern),
                User.anonymous_handle.ilike(pattern),
                User.display_name.ilike(pattern),
            )
        )

    rows = query.order_by(User.created_at.desc()).limit(limit).all()

    row_html: list[str] = []
    for user, decommission in rows:
        account_chip = (
            "<span class='chip user-active'>active</span>"
            if decommission is None
            else "<span class='chip user-decommissioned'>decommissioned</span>"
        )
        display_name = user.display_name or "-"
        reason = _snippet(decommission.reason if decommission is not None else None, 80)
        decommissioned_at = _fmt_dt(decommission.decommissioned_at if decommission is not None else None)

        if decommission is None:
            action_html = (
                f"<form class='inline-form' method='post' action='/admin/users/{escape(user.id)}/decommission'>"
                "<input type='hidden' name='redirect_to' value='/admin/users' />"
                "<input type='text' name='reason' placeholder='Reason (optional)' maxlength='255' />"
                "<button class='danger' type='submit'>Decommission</button>"
                "</form>"
            )
        else:
            action_html = (
                f"<form class='inline-form' method='post' action='/admin/users/{escape(user.id)}/recommission'>"
                "<input type='hidden' name='redirect_to' value='/admin/users' />"
                "<button class='ok' type='submit'>Recommission</button>"
                "</form>"
            )

        row_html.append(
            "<tr>"
            f"<td>{escape(user.id[:8])}</td>"
            f"<td>@{escape(user.anonymous_handle)}</td>"
            f"<td>{escape(display_name)}</td>"
            f"<td>{account_chip}</td>"
            f"<td>{escape(reason)}</td>"
            f"<td>{escape(decommissioned_at)}</td>"
            f"<td>{action_html}</td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr><th>User</th><th>Handle</th><th>Display Name</th><th>Status</th><th>Reason</th><th>Decommissioned At</th><th>Actions</th></tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        if row_html
        else "<p class='empty'>No users matched your filters.</p>"
    )

    selected_active = "selected" if account_state == "active" else ""
    selected_decommissioned = "selected" if account_state == "decommissioned" else ""
    selected_all = "selected" if account_state == "all" else ""

    body = f"""
      <h1>Account Lifecycle</h1>
      <form method=\"get\">
        <input type=\"text\" name=\"q\" value=\"{escape(q)}\" placeholder=\"Search by id / handle / name / apple_sub\" />
        <select name=\"account_state\">
          <option value=\"active\" {selected_active}>Active</option>
          <option value=\"decommissioned\" {selected_decommissioned}>Decommissioned</option>
          <option value=\"all\" {selected_all}>All</option>
        </select>
        <input type=\"number\" name=\"limit\" min=\"1\" max=\"300\" value=\"{limit}\" />
        <button type=\"submit\">Apply</button>
      </form>
      {table_html}
    """

    return _layout("Users", body)


@router.get("/feedback", response_class=HTMLResponse)
def admin_feedback(
    q: str = Query(default="", max_length=200),
    kind: str = Query(default="all", pattern="^(all|idea|bug)$"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_admin_db),
    _: None = Depends(require_admin),
) -> HTMLResponse:
    query = (
        db.query(FeedbackReport, User)
        .join(User, User.id == FeedbackReport.user_id)
    )

    if kind != "all":
        query = query.filter(FeedbackReport.kind == kind)

    if q.strip():
        pattern = f"%{q.strip()}%"
        query = query.filter(
            or_(
                FeedbackReport.message.ilike(pattern),
                FeedbackReport.user_id.ilike(pattern),
                User.anonymous_handle.ilike(pattern),
                User.display_name.ilike(pattern),
            )
        )

    rows = (
        query.order_by(FeedbackReport.created_at.desc(), FeedbackReport.id.desc())
        .limit(limit)
        .all()
    )

    idea_count = (
        db.query(func.count(FeedbackReport.id))
        .filter(FeedbackReport.kind == "idea")
        .scalar()
        or 0
    )
    bug_count = (
        db.query(func.count(FeedbackReport.id))
        .filter(FeedbackReport.kind == "bug")
        .scalar()
        or 0
    )

    row_html: list[str] = []
    for report, user in rows:
        display_name = user.display_name or "-"
        row_html.append(
            "<tr>"
            f"<td>{escape(user.id[:8])}</td>"
            f"<td>@{escape(user.anonymous_handle)}</td>"
            f"<td>{escape(display_name)}</td>"
            f"<td><span class='chip'>{escape(report.kind)}</span></td>"
            f"<td>{escape(_snippet(report.message, 260))}</td>"
            f"<td>{escape(_fmt_dt(report.created_at))}</td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr><th>User</th><th>Handle</th><th>Display Name</th><th>Type</th><th>Message</th><th>Created</th></tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        if row_html
        else "<p class='empty'>No feedback matched your filters.</p>"
    )

    selected_all = "selected" if kind == "all" else ""
    selected_idea = "selected" if kind == "idea" else ""
    selected_bug = "selected" if kind == "bug" else ""

    body = f"""
      <h1>Feedback Reports</h1>
      <p class=\"meta\">Idea and bug report submissions from in-app settings.</p>
      <section class=\"cards\">
        <div class=\"card\"><div class=\"label\">Rows</div><div class=\"value\">{len(rows)}</div></div>
        <div class=\"card\"><div class=\"label\">Ideas</div><div class=\"value\">{idea_count}</div></div>
        <div class=\"card\"><div class=\"label\">Bugs</div><div class=\"value\">{bug_count}</div></div>
      </section>
      <form method=\"get\">
        <input type=\"text\" name=\"q\" value=\"{escape(q)}\" placeholder=\"Search message / user / handle\" />
        <select name=\"kind\">
          <option value=\"all\" {selected_all}>All</option>
          <option value=\"idea\" {selected_idea}>Idea</option>
          <option value=\"bug\" {selected_bug}>Bug</option>
        </select>
        <input type=\"number\" name=\"limit\" min=\"1\" max=\"500\" value=\"{limit}\" />
        <button type=\"submit\">Apply</button>
      </form>
      {table_html}
    """

    return _layout("Feedback", body)


@router.post("/users/{user_id}/decommission")
async def admin_decommission_user(
    user_id: str,
    request: Request,
    db: Session = Depends(get_admin_db),
    _: None = Depends(require_admin),
) -> RedirectResponse:
    user = db.query(User).filter(User.id == user_id).one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    form = await request.form()
    reason = str(form.get("reason") or "").strip() or None
    redirect_to = _safe_redirect_target(str(form.get("redirect_to") or ""), fallback="/admin/users")

    decommission = db.query(AccountDecommission).filter(AccountDecommission.user_id == user.id).one_or_none()
    if decommission is None:
        decommission = AccountDecommission(user_id=user.id, reason=reason, decommissioned_at=now_utc())
    else:
        decommission.reason = reason
        decommission.decommissioned_at = now_utc()
    db.add(decommission)
    db.commit()

    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/users/{user_id}/recommission")
async def admin_recommission_user(
    user_id: str,
    request: Request,
    db: Session = Depends(get_admin_db),
    _: None = Depends(require_admin),
) -> RedirectResponse:
    user = db.query(User).filter(User.id == user_id).one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    form = await request.form()
    redirect_to = _safe_redirect_target(str(form.get("redirect_to") or ""), fallback="/admin/users")

    decommission = db.query(AccountDecommission).filter(AccountDecommission.user_id == user.id).one_or_none()
    if decommission is not None:
        db.delete(decommission)
        db.commit()

    return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/entries/{entry_id}", response_class=HTMLResponse)
def admin_entry_detail(
    entry_id: str,
    db: Session = Depends(get_admin_db),
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
