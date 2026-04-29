import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import html
import hmac
import json
import os
import time
import uuid
from datetime import UTC, datetime

import jwt
from fastapi import Cookie, Depends, FastAPI, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import analysis, auth, db, schemas


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(db.init_db)
    yield


app = FastAPI(title="blackhole", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ADMIN_COOKIE_NAME = "blackhole_admin_session"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_SESSION_HOURS = 24 * 30
ADMIN_ITEM_LIMIT = int(os.getenv("ADMIN_ITEM_LIMIT", "2000"))


async def get_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    try:
        return auth.verify_session_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired session token")


def _create_admin_session(username: str) -> str:
    now = int(time.time())
    payload = {
        "sub": username,
        "scope": "admin",
        "iat": now,
        "exp": now + ADMIN_SESSION_HOURS * 3600,
    }
    return jwt.encode(payload, auth.JWT_SECRET, algorithm=auth.JWT_ALGORITHM)


def _verify_admin_session(token: str | None) -> str:
    if not token:
        raise HTTPException(401, "Missing admin session")

    try:
        payload = jwt.decode(token, auth.JWT_SECRET, algorithms=[auth.JWT_ALGORITHM])
    except Exception:
        raise HTTPException(401, "Invalid admin session")

    if payload.get("scope") != "admin":
        raise HTTPException(403, "Invalid admin scope")

    return payload["sub"]


async def require_admin(admin_session: str | None = Cookie(None, alias=ADMIN_COOKIE_NAME)) -> str:
    return _verify_admin_session(admin_session)


@app.post("/auth/apple")
async def sign_in_apple(body: schemas.AppleSignInRequest):
    try:
        apple_user_id = await auth.verify_apple_token(body.identity_token)
    except Exception as e:
        raise HTTPException(401, f"Apple token verification failed: {e}")

    await asyncio.to_thread(db.upsert_user, apple_user_id)
    session_token = auth.create_session_token(apple_user_id)
    return {"session_token": session_token}


@app.post("/items", response_model=list[schemas.Item])
async def create_item(body: schemas.CreateItemRequest, user_id: str = Depends(get_user_id)):
    prior_rows = await asyncio.to_thread(db.list_items_recent, user_id, 25)
    epic_rows = await asyncio.to_thread(db.list_epics, user_id)
    prior_items = _merge_context_rows(epic_rows, prior_rows)
    results, llm_log = await asyncio.to_thread(
        analysis.run_transcript_analysis, body.content, prior_items
    )
    if not results:
        results = [{"type": "note", "title": body.content[:60], "tags": [], "due_date": None}]

    now = datetime.now(UTC).isoformat()
    created = []
    epic_lookup = _epic_lookup([dict(row) for row in epic_rows])
    prepared_results = []
    for result in results:
        item_id = str(uuid.uuid4())
        prepared_results.append((item_id, result))
        if result.get("type") == "epic":
            epic_lookup[_epic_key(result.get("title", ""))] = item_id

    for item_id, result in prepared_results:
        item_type = result.get("type", "note")
        epic_title = result.get("epic_title")
        epic_id = None if item_type == "epic" else _resolve_epic_id(epic_title, epic_lookup)
        tags = result.get("tags", [])
        if epic_id and epic_title:
            tags = _tags_with_epic(tags, epic_title)

        item_data = {
            "id": item_id,
            "user_id": user_id,
            "content": body.content,
            "title": result.get("title", body.content[:60]),
            "type": item_type,
            "epic_id": epic_id,
            "due_date": result.get("due_date"),
            "completed": 0,
            "tags": json.dumps(tags),
            "created_at": now,
            "updated_at": now,
        }
        await asyncio.to_thread(db.create_item, item_data)
        created.append(item_data)

    await _persist_llm_log(llm_log, user_id=user_id, item_id=created[0]["id"] if created else None, created_at=now)

    return [_to_schema(item) for item in created]


@app.get("/items", response_model=list[schemas.Item])
async def list_items(user_id: str = Depends(get_user_id)):
    rows = await asyncio.to_thread(db.list_items, user_id)
    return [_to_schema(dict(r)) for r in rows]


@app.patch("/items/{item_id}", response_model=schemas.Item)
async def update_item(
    item_id: str,
    body: schemas.UpdateItemRequest,
    user_id: str = Depends(get_user_id),
):
    row = await asyncio.to_thread(db.get_item, item_id, user_id)
    if not row:
        raise HTTPException(404, "Item not found")

    provided = body.model_fields_set
    updates: dict = {}
    if "completed" in provided:
        updates["completed"] = int(body.completed)
    if "title" in provided and body.title is not None:
        updates["title"] = body.title
    if "content" in provided and body.content is not None:
        updates["content"] = body.content
    if "type" in provided and body.type is not None:
        updates["type"] = body.type
        if body.type == "epic":
            updates["epic_id"] = None
    epic_title_for_tag = None
    if "epic_id" in provided:
        if body.epic_id:
            epic = await asyncio.to_thread(db.get_item, body.epic_id, user_id)
            if not epic or epic["type"] != "epic":
                raise HTTPException(400, "Epic not found")
            updates["epic_id"] = body.epic_id
            epic_title_for_tag = epic["title"]
        else:
            updates["epic_id"] = None
    if "due_date" in provided:
        updates["due_date"] = body.due_date  # None clears it
    if "tags" in provided and body.tags is not None:
        updates["tags"] = json.dumps(body.tags)
    if epic_title_for_tag:
        base_tags = updates.get("tags", row["tags"])
        if isinstance(base_tags, str):
            try:
                base_tags = json.loads(base_tags)
            except Exception:
                base_tags = []
        updates["tags"] = json.dumps(_tags_with_epic(base_tags, epic_title_for_tag))
    updates["updated_at"] = datetime.now(UTC).isoformat()

    await asyncio.to_thread(db.update_item, item_id, updates)
    updated = await asyncio.to_thread(db.get_item, item_id, user_id)
    return _to_schema(dict(updated))


@app.delete("/items/{item_id}")
async def delete_item(item_id: str, user_id: str = Depends(get_user_id)):
    row = await asyncio.to_thread(db.get_item, item_id, user_id)
    if not row:
        raise HTTPException(404, "Item not found")
    await asyncio.to_thread(db.delete_item, item_id)
    return {}


@app.post("/search", response_model=list[schemas.Item])
async def search(body: schemas.SearchRequest, user_id: str = Depends(get_user_id)):
    rows = await asyncio.to_thread(db.list_items, user_id)
    if not rows:
        return []

    items = [dict(r) for r in rows]
    results, llm_log = await asyncio.to_thread(analysis.run_search_items, body.query, items)
    await _persist_llm_log(llm_log, user_id=user_id, item_id=None)

    if results is None:
        q = body.query.lower()
        results = [i for i in items if _item_matches_query(i, q)]

    return [_to_schema(i) for i in results]


def _item_matches_query(item: dict, query: str) -> bool:
    tags = item.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

    fields = [
        item.get("title", ""),
        item.get("content", ""),
        item.get("type", ""),
        item.get("epic_id", ""),
        item.get("due_date", ""),
        "completed" if item.get("completed") else "open",
        *tags,
    ]
    return query in " ".join(str(field).lower() for field in fields)


def _merge_context_rows(*row_groups: list) -> list[dict]:
    merged = []
    seen = set()
    for rows in row_groups:
        for row in rows:
            item = dict(row)
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            merged.append(item)
    return merged


def _epic_lookup(epics: list[dict]) -> dict[str, str]:
    return {
        key: epic["id"]
        for epic in epics
        if (key := _epic_key(epic.get("title", "")))
    }


def _epic_key(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(str(title).strip().lower().split())


def _resolve_epic_id(epic_title: str | None, epic_lookup: dict[str, str]) -> str | None:
    return epic_lookup.get(_epic_key(epic_title))


def _tags_with_epic(tags: list | None, epic_title: str) -> list[str]:
    normalized_existing = {str(tag).strip().lower() for tag in tags or []}
    result = [str(tag) for tag in tags or [] if str(tag).strip()]
    title = epic_title.strip()
    if title and title.lower() not in normalized_existing:
        result.append(title)
    return result


async def _persist_llm_log(
    llm_log: dict,
    *,
    user_id: str,
    item_id: str | None,
    created_at: str | None = None,
) -> None:
    llm_log.update(
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": item_id,
            "created_at": created_at or datetime.now(UTC).isoformat(),
        }
    )
    try:
        await asyncio.to_thread(db.create_llm_log, llm_log)
    except Exception:
        pass


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/admin", status_code=302)


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard(admin_session: str | None = Cookie(None, alias=ADMIN_COOKIE_NAME)):
    if not admin_session:
        return HTMLResponse(_render_admin_login())

    try:
        username = _verify_admin_session(admin_session)
    except HTTPException:
        response = HTMLResponse(_render_admin_login(error="Your admin session expired. Sign in again."))
        response.delete_cookie(ADMIN_COOKIE_NAME)
        return response

    return HTMLResponse(_render_admin_dashboard(username, _admin_payload()))


@app.post("/admin/login", include_in_schema=False)
async def admin_login(username: str = Form(...), password: str = Form(...)):
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        return HTMLResponse(_render_admin_login(error="Admin credentials are not configured on the server."), status_code=503)

    username_ok = hmac.compare_digest(username, ADMIN_USERNAME)
    password_ok = hmac.compare_digest(password, ADMIN_PASSWORD)
    if not (username_ok and password_ok):
        return HTMLResponse(_render_admin_login(error="Incorrect username or password."), status_code=401)

    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=_create_admin_session(username),
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=ADMIN_SESSION_HOURS * 3600,
    )
    return response


@app.post("/admin/logout", include_in_schema=False)
async def admin_logout():
    response = RedirectResponse(url="/admin", status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return response


@app.get("/admin/api/overview", response_class=JSONResponse, include_in_schema=False)
async def admin_overview(_: str = Depends(require_admin)):
    return JSONResponse(_admin_payload())


def _to_schema(item: dict) -> schemas.Item:
    tags = item.get("tags", "[]")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

    return schemas.Item(
        id=item["id"],
        content=item["content"],
        title=item["title"],
        type=item["type"],
        epic_id=item.get("epic_id"),
        due_date=item.get("due_date"),
        completed=bool(item.get("completed", 0)),
        tags=tags,
        created_at=item["created_at"],
        updated_at=item["updated_at"],
    )


def _admin_payload() -> dict:
    summary = db.get_admin_summary()
    llm_summary = db.get_llm_log_summary()
    users = [dict(row) for row in db.list_all_users()]
    items = [_serialize_admin_item(dict(row)) for row in db.list_all_items(limit=ADMIN_ITEM_LIMIT)]
    llm_logs = [dict(row) for row in db.list_llm_logs(limit=200)]
    recent_items = items[:100]
    due_items = [_serialize_admin_item(dict(row)) for row in summary["due_items"]]
    tags = Counter()
    now = datetime.now(UTC)
    active_users_7d = set()
    overdue_items = 0
    items_created_7d = 0

    for item in items:
        for tag in item["tags"]:
            tags[tag] += 1

        created_at = _parse_timestamp(item.get("created_at"))
        updated_at = _parse_timestamp(item.get("updated_at"))
        due_at = _parse_timestamp(item.get("due_date"))

        if created_at and (now - created_at).days < 7:
            items_created_7d += 1
        if updated_at and (now - updated_at).days < 7:
            active_users_7d.add(item["user_id"])
        if due_at and due_at < now and not item["completed"]:
            overdue_items += 1

    for user in users:
        user["open_count"] = int(user["item_count"]) - int(user["completed_count"])

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            **summary,
            "due_items": due_items,
            "returned_items": len(items),
            "item_limit": ADMIN_ITEM_LIMIT,
            "inventory_truncated": summary["total_items"] > len(items),
            "overdue_items": overdue_items,
            "active_users_7d": len(active_users_7d),
            "items_created_7d": items_created_7d,
            "completion_rate": round(
                (summary["completed_items"] / summary["total_items"]) * 100, 1
            ) if summary["total_items"] else 0.0,
            "top_tags": [
                {"tag": tag, "count": count}
                for tag, count in tags.most_common(12)
            ],
            "llm_total_logs": llm_summary["total_logs"],
            "llm_failed_logs": llm_summary["failed_logs"],
            "llm_last_log_at": llm_summary["last_log_at"],
            "llm_operations": llm_summary["operations"],
        },
        "users": users,
        "items": items,
        "recent_items": recent_items,
        "llm_logs": llm_logs,
    }


def _serialize_admin_item(item: dict) -> dict:
    tags = item.get("tags", "[]")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []
    item["tags"] = tags
    item["completed"] = bool(item.get("completed", 0))
    return item


def _render_admin_login(error: str | None = None) -> str:
    error_html = (
        f'<div class="alert">{html.escape(error)}</div>'
        if error
        else '<p class="subtle">Sign in to inspect every user and item currently stored in blackhole.</p>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>blackhole admin</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --line: #d9dde3;
      --text: #111418;
      --muted: #66707d;
      --accent: #111418;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef2f6 0%, var(--bg) 100%);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    .card {{
      width: min(420px, 100%);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(17, 20, 24, 0.08);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0 0 18px; color: var(--muted); line-height: 1.5; }}
    label {{ display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; }}
    input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 15px;
      margin-bottom: 14px;
      background: #fff;
    }}
    button {{
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 13px 14px;
      background: var(--accent);
      color: #fff;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }}
    .alert {{
      color: var(--danger);
      background: #fef3f2;
      border: 1px solid #fecdca;
      padding: 12px 14px;
      border-radius: 12px;
      margin-bottom: 18px;
      font-size: 14px;
    }}
    .subtle {{ font-size: 14px; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>blackhole admin</h1>
    {error_html}
    <form method="post" action="/admin/login">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" required />
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required />
      <button type="submit">Sign In</button>
    </form>
  </main>
</body>
</html>"""


def _render_admin_dashboard(username: str, payload: dict) -> str:
    summary = payload["summary"]
    users = payload["users"]
    items = payload["items"]
    recent_items = payload["recent_items"]
    llm_logs = payload["llm_logs"]
    due_items = summary["due_items"]
    type_rows = summary["types"]
    top_tags = summary["top_tags"]
    llm_operations = summary["llm_operations"]
    daily_rows = list(reversed(summary["daily_activity"]))
    max_daily = max((row["count"] for row in daily_rows), default=1)
    max_tag_count = max((row["count"] for row in top_tags), default=1)
    items_json = json.dumps(items).replace("</", "<\\/")

    metric_cards = [
        ("Users", str(summary["total_users"])),
        ("Items", str(summary["total_items"])),
        ("Open", str(summary["open_items"])),
        ("Completed", str(summary["completed_items"])),
        ("Overdue", str(summary["overdue_items"])),
        ("Active Users / 7d", str(summary["active_users_7d"])),
        ("New Items / 7d", str(summary["items_created_7d"])),
        ("LLM Calls", str(summary["llm_total_logs"])),
        ("LLM Failures", str(summary["llm_failed_logs"])),
        ("Completion Rate", f'{summary["completion_rate"]}%'),
        ("Refresh", _format_timestamp(payload["generated_at"])),
    ]

    metrics_html = "".join(
        f"""
        <section class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value">{html.escape(value)}</div>
        </section>
        """
        for label, value in metric_cards
    )

    users_html = "".join(
        f"""
        <tr>
          <td class="mono">{html.escape(user["id"])}</td>
          <td>{int(user["item_count"])}</td>
          <td>{int(user["open_count"])}</td>
          <td>{int(user["completed_count"])}</td>
          <td>{html.escape(_format_timestamp(user["last_activity_at"]))}</td>
          <td>{html.escape(_format_timestamp(user["created_at"]))}</td>
        </tr>
        """
        for user in users
    ) or '<tr><td colspan="6" class="empty">No users yet.</td></tr>'

    recent_items_html = "".join(_render_item_row(item) for item in recent_items) or (
        '<tr><td colspan="8" class="empty">No items yet.</td></tr>'
    )
    due_items_html = "".join(_render_item_row(item, include_due_focus=True) for item in due_items) or (
        '<tr><td colspan="8" class="empty">No due-dated items.</td></tr>'
    )

    type_html = "".join(
        f"""
        <div class="breakdown-row">
          <span>{html.escape(row["type"] or "unknown")}</span>
          <strong>{row["count"]}</strong>
        </div>
        """
        for row in type_rows
    ) or '<div class="empty-panel">No item types recorded yet.</div>'

    activity_html = "".join(
        f"""
        <div class="activity-row">
          <div class="activity-meta">
            <span>{html.escape(row["day"])}</span>
            <strong>{row["count"]}</strong>
          </div>
          <div class="activity-bar"><span style="width: {max(8, int((row["count"] / max_daily) * 100))}%"></span></div>
        </div>
        """
        for row in daily_rows
    ) or '<div class="empty-panel">No recent activity yet.</div>'

    tag_html = "".join(
        f"""
        <div class="activity-row">
          <div class="activity-meta">
            <span>{html.escape(row["tag"])}</span>
            <strong>{row["count"]}</strong>
          </div>
          <div class="activity-bar"><span style="width: {max(8, int((row["count"] / max_tag_count) * 100))}%"></span></div>
        </div>
        """
        for row in top_tags
    ) or '<div class="empty-panel">No tags recorded yet.</div>'

    llm_operation_html = "".join(
        f"""
        <div class="breakdown-row">
          <span>{html.escape(row["operation"])}</span>
          <strong>{row["count"]}</strong>
        </div>
        """
        for row in llm_operations
    ) or '<div class="empty-panel">No LLM calls logged yet.</div>'

    llm_logs_html = "".join(_render_llm_log_row(log) for log in llm_logs) or (
        '<tr><td colspan="7" class="empty">No LLM logs yet.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>blackhole admin</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f4f6;
      --panel: #ffffff;
      --panel-alt: #f8fafc;
      --line: #dde2e8;
      --text: #111418;
      --muted: #66707d;
      --accent: #16181d;
      --ok: #0f766e;
      --warn: #b54708;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(191, 219, 254, 0.35), transparent 26%),
        linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1440px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      letter-spacing: -0.03em;
    }}
    .subtle {{ color: var(--muted); font-size: 14px; }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .button, button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 12px;
      padding: 10px 14px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
    }}
    .button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .metrics {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 16px;
    }}
    .metric-card, .panel {{
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(17, 20, 24, 0.05);
    }}
    .metric-card {{
      padding: 18px;
      min-height: 108px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .metric-label {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ font-size: 30px; font-weight: 800; letter-spacing: -0.04em; }}
    .two-up {{
      grid-template-columns: 1.2fr 0.8fr;
      margin-bottom: 16px;
    }}
    .three-up {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .panel {{
      padding: 18px;
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-alt);
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1.4fr repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .control label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .control input, .control select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #fbfcfd;
      white-space: nowrap;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
      font-size: 12px;
    }}
    .tags {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .tag, .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .pill.ok {{
      color: var(--ok);
      border-color: rgba(15, 118, 110, 0.18);
      background: rgba(15, 118, 110, 0.08);
    }}
    .pill.warn {{
      color: var(--warn);
      border-color: rgba(181, 71, 8, 0.18);
      background: rgba(181, 71, 8, 0.08);
    }}
    .breakdown-row, .activity-row {{
      display: grid;
      gap: 10px;
      padding: 10px 0;
    }}
    .breakdown-row {{
      grid-template-columns: 1fr auto;
      border-bottom: 1px solid var(--line);
    }}
    .breakdown-row:last-child {{ border-bottom: 0; }}
    .activity-meta {{
      display: flex;
      justify-content: space-between;
      font-size: 13px;
    }}
    .activity-bar {{
      height: 10px;
      border-radius: 999px;
      background: #e8edf3;
      overflow: hidden;
    }}
    .activity-bar span {{
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #111418 0%, #6b7280 100%);
    }}
    .empty, .empty-panel {{
      color: var(--muted);
      text-align: center;
      padding: 18px;
    }}
    .content-cell {{
      min-width: 260px;
      max-width: 380px;
    }}
    .content-cell strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
    }}
    .content-cell p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .inventory-note {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 8px 10px;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--text);
    }}
    pre {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
      font-size: 12px;
      line-height: 1.5;
      color: var(--muted);
    }}
    .status-stack {{
      display: grid;
      gap: 8px;
    }}
    @media (max-width: 960px) {{
      .two-up {{ grid-template-columns: 1fr; }}
      .three-up {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
      .page {{ width: min(100%, calc(100% - 20px)); }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="topbar">
      <div>
        <h1>blackhole admin</h1>
        <div class="subtle">Signed in as {html.escape(username)}. Inventory shows {len(items)} items{", truncated by server limit" if summary["inventory_truncated"] else ""}.</div>
      </div>
      <div class="actions">
        <a class="button primary" href="/admin/api/overview" target="_blank" rel="noreferrer">Open JSON</a>
        <a class="button" href="/admin">Refresh</a>
        <form method="post" action="/admin/logout">
          <button type="submit">Log Out</button>
        </form>
      </div>
    </section>

    <section class="grid metrics">{metrics_html}</section>

    <section class="grid two-up">
      <section class="panel">
        <div class="panel-header">
          <h2>Recent Item Activity</h2>
          <span class="subtle">Last 100 items across all users</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>User</th>
                <th>Type</th>
                <th>Status</th>
                <th>Due</th>
                <th>Tags</th>
                <th>Content</th>
                <th>Created</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>{recent_items_html}</tbody>
          </table>
        </div>
      </section>

      <section class="grid">
        <section class="panel">
          <h2>Type Breakdown</h2>
          {type_html}
        </section>
        <section class="panel">
          <h2>Daily Activity</h2>
          {activity_html}
        </section>
        <section class="panel">
          <h2>Top Tags</h2>
          {tag_html}
        </section>
      </section>
    </section>

    <section class="grid two-up">
      <section class="panel">
        <div class="panel-header">
          <h2>LLM Analysis Logs</h2>
          <span class="subtle">Raw ingest calls and raw model outputs</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Logged</th>
                <th>Status</th>
                <th>User</th>
                <th>Item</th>
                <th>Model</th>
                <th>Transcript</th>
                <th>Prompts / Response</th>
              </tr>
            </thead>
            <tbody>{llm_logs_html}</tbody>
          </table>
        </div>
      </section>

      <section class="grid">
        <section class="panel">
          <h2>LLM Operations</h2>
          {llm_operation_html}
        </section>
        <section class="panel">
          <h2>Last LLM Log</h2>
          <div class="subtle">{html.escape(_format_timestamp(summary["llm_last_log_at"]))}</div>
        </section>
      </section>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Full Inventory</h2>
        <span class="subtle">Search and filter all returned items</span>
      </div>
      <div class="controls">
        <div class="control">
          <label for="inventory-search">Search</label>
          <input id="inventory-search" type="search" placeholder="Title, content, tags, user id" />
        </div>
        <div class="control">
          <label for="inventory-user">User</label>
          <select id="inventory-user"><option value="">All users</option></select>
        </div>
        <div class="control">
          <label for="inventory-type">Type</label>
          <select id="inventory-type"><option value="">All types</option></select>
        </div>
        <div class="control">
          <label for="inventory-status">Status</label>
          <select id="inventory-status">
            <option value="">All statuses</option>
            <option value="open">Open</option>
            <option value="completed">Completed</option>
          </select>
        </div>
        <div class="control">
          <label for="inventory-due">Due</label>
          <select id="inventory-due">
            <option value="">Any due state</option>
            <option value="overdue">Overdue</option>
            <option value="dated">Has due date</option>
            <option value="none">No due date</option>
          </select>
        </div>
      </div>
      <div class="inventory-note">
        <span id="inventory-count">Showing 0 items</span>
        <span>Server cap: {summary["item_limit"]} items</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>User</th>
              <th>Type</th>
              <th>Status</th>
              <th>Due</th>
              <th>Tags</th>
              <th>Content</th>
              <th>Created</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody id="inventory-body"></tbody>
        </table>
      </div>
    </section>

    <section class="grid two-up">
      <section class="panel">
        <div class="panel-header">
          <h2>Users</h2>
          <span class="subtle">All known accounts</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>User ID</th>
                <th>Items</th>
                <th>Open</th>
                <th>Completed</th>
                <th>Last Activity</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>{users_html}</tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>Upcoming / Dated Items</h2>
          <span class="subtle">Earliest due dates first</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>User</th>
                <th>Type</th>
                <th>Status</th>
                <th>Due</th>
                <th>Tags</th>
                <th>Content</th>
                <th>Created</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>{due_items_html}</tbody>
          </table>
        </div>
      </section>
    </section>
  </main>
  <script>
    const inventoryItems = {items_json};
    const inventoryBody = document.getElementById("inventory-body");
    const inventoryCount = document.getElementById("inventory-count");
    const searchInput = document.getElementById("inventory-search");
    const userSelect = document.getElementById("inventory-user");
    const typeSelect = document.getElementById("inventory-type");
    const statusSelect = document.getElementById("inventory-status");
    const dueSelect = document.getElementById("inventory-due");
    const now = new Date();

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }})[char]);
    }}

    function itemDueState(item) {{
      if (!item.due_date) return "none";
      const dueAt = new Date(item.due_date);
      if (!Number.isNaN(dueAt.valueOf()) && dueAt < now && !item.completed) return "overdue";
      return "dated";
    }}

    function rowHtml(item) {{
      const tags = item.tags.length
        ? item.tags.map((tag) => `<span class="tag">${{escapeHtml(tag)}}</span>`).join("")
        : '<span class="subtle">—</span>';
      const statusClass = item.completed ? "ok" : "warn";
      const statusLabel = item.completed ? "Completed" : "Open";
      const dueValue = item.due_date || "—";
      return `
        <tr>
          <td class="mono">${{escapeHtml(item.user_id)}}</td>
          <td><span class="pill">${{escapeHtml(item.type || "note")}}</span></td>
          <td><span class="pill ${{statusClass}}">${{statusLabel}}</span></td>
          <td>${{escapeHtml(dueValue)}}</td>
          <td><div class="tags">${{tags}}</div></td>
          <td class="content-cell">
            <strong>${{escapeHtml(item.title || "Untitled")}}</strong>
            <p>${{escapeHtml(item.content || "")}}</p>
          </td>
          <td>${{escapeHtml(item.created_at || "—")}}</td>
          <td>${{escapeHtml(item.updated_at || "—")}}</td>
        </tr>
      `;
    }}

    function renderInventory() {{
      const search = searchInput.value.trim().toLowerCase();
      const user = userSelect.value;
      const type = typeSelect.value;
      const status = statusSelect.value;
      const due = dueSelect.value;

      const filtered = inventoryItems.filter((item) => {{
        const haystack = [item.title, item.content, item.user_id, ...(item.tags || [])].join(" ").toLowerCase();
        if (search && !haystack.includes(search)) return false;
        if (user && item.user_id !== user) return false;
        if (type && item.type !== type) return false;
        if (status === "open" && item.completed) return false;
        if (status === "completed" && !item.completed) return false;
        if (due && itemDueState(item) !== due) return false;
        return true;
      }});

      inventoryCount.textContent = `Showing ${{filtered.length}} of ${{inventoryItems.length}} returned items`;
      inventoryBody.innerHTML = filtered.length
        ? filtered.map(rowHtml).join("")
        : '<tr><td colspan="8" class="empty">No items match the current filters.</td></tr>';
    }}

    function populateOptions(select, values, label) {{
      values.forEach((value) => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value || label;
        select.appendChild(option);
      }});
    }}

    populateOptions(userSelect, [...new Set(inventoryItems.map((item) => item.user_id))].sort(), "All users");
    populateOptions(typeSelect, [...new Set(inventoryItems.map((item) => item.type || "note"))].sort(), "All types");

    [searchInput, userSelect, typeSelect, statusSelect, dueSelect].forEach((element) => {{
      element.addEventListener("input", renderInventory);
      element.addEventListener("change", renderInventory);
    }});

    renderInventory();
  </script>
</body>
</html>"""


def _render_item_row(item: dict, include_due_focus: bool = False) -> str:
    due_value = item.get("due_date") or "—"
    tags_html = "".join(
        f'<span class="tag">{html.escape(tag)}</span>' for tag in item.get("tags", [])
    ) or '<span class="subtle">—</span>'
    content_preview = html.escape(item.get("content", ""))
    title = html.escape(item.get("title", "Untitled"))
    status_class = "ok" if item.get("completed") else "warn"
    status_label = "Completed" if item.get("completed") else "Open"
    due_display = html.escape(due_value)
    if include_due_focus and due_value != "—":
        due_display = f"<strong>{due_display}</strong>"

    return f"""
    <tr>
      <td class="mono">{html.escape(item.get("user_id", ""))}</td>
      <td><span class="pill">{html.escape(item.get("type", "note"))}</span></td>
      <td><span class="pill {status_class}">{status_label}</span></td>
      <td>{due_display}</td>
      <td><div class="tags">{tags_html}</div></td>
      <td class="content-cell">
        <strong>{title}</strong>
        <p>{content_preview}</p>
      </td>
      <td>{html.escape(_format_timestamp(item.get("created_at")))}</td>
      <td>{html.escape(_format_timestamp(item.get("updated_at")))}</td>
    </tr>
    """


def _render_llm_log_row(log: dict) -> str:
    status_class = "warn" if log.get("status") == "error" else "ok"
    status_label = (log.get("status") or "unknown").capitalize()

    prompt_parts = [
        _render_log_block("System Prompt", log.get("system_prompt")),
        _render_log_block("User Prompt", log.get("user_prompt")),
        _render_log_block("Raw Response", log.get("raw_response")),
        _render_log_block("Parsed Response", log.get("parsed_response")),
        _render_log_block("Error", log.get("error")),
    ]
    prompt_html = "".join(part for part in prompt_parts if part)

    return f"""
    <tr>
      <td>{html.escape(_format_timestamp(log.get("created_at")))}</td>
      <td>
        <div class="status-stack">
          <span class="pill {status_class}">{html.escape(status_label)}</span>
          <span class="mono">{html.escape(log.get("operation", "unknown"))}</span>
        </div>
      </td>
      <td class="mono">{html.escape(log.get("user_id") or "—")}</td>
      <td class="mono">{html.escape(log.get("item_id") or "—")}</td>
      <td class="mono">{html.escape(log.get("model") or "—")}</td>
      <td>{_render_log_block("Transcript", log.get("input_text"), open_by_default=True)}</td>
      <td>{prompt_html or '<span class="subtle">—</span>'}</td>
    </tr>
    """


def _render_log_block(label: str, value: str | None, open_by_default: bool = False) -> str:
    if not value:
        return ""

    open_attr = " open" if open_by_default else ""
    return (
        f'<details{open_attr}><summary>{html.escape(label)}</summary>'
        f'<pre>{html.escape(value)}</pre></details>'
    )


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "—"

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return value

    return parsed.strftime("%Y-%m-%d %H:%M")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except Exception:
            continue

    return None
