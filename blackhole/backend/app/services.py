import hashlib
import json
import re
import secrets
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from . import db

ITEM_TYPES = {
    "note",
    "todo",
    "event",
    "epic",
    "contact",
    "resource",
    "decision",
    "journal",
    "habit",
    "table",
}
PUBLIC_ITEM_TYPES = ITEM_TYPES - {"table"}
STATUSES = {"open", "in_progress", "done", "cancelled", "archived"}
PRIORITIES = {"low", "medium", "high", "urgent"}
SOURCES = {"voice", "manual", "agent", "import", "system"}
READ_STATUSES = {"unread", "reading", "read", "skipped"}
LINK_TYPES = {"reference", "blocks", "spawned_from", "relates_to", "contradicts"}
TABLE_COLUMN_TYPES = {"text", "number", "boolean", "date", "select"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PATCHABLE_DECISION_FIELDS = {"tags", "parent_id", "epic_id", "status", "completed", "priority", "metadata"}


class ServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def row_to_item(row) -> dict:
    item = dict(row)
    status = item.get("status") or ("done" if item.get("completed") else "open")
    parent_id = item.get("parent_id") or item.get("epic_id")
    return {
        "id": item["id"],
        "type": item.get("type") or "note",
        "title": item.get("title") or "",
        "content": item.get("content") or "",
        "status": status,
        "priority": item.get("priority"),
        "source": item.get("source") or "voice",
        "parent_id": parent_id,
        "epic_id": item.get("epic_id"),
        "completed": status == "done",
        "tags": _json_list(item.get("tags")),
        "due_date": item.get("due_date"),
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
        "location": item.get("location"),
        "url": item.get("url"),
        "read_status": item.get("read_status"),
        "email": item.get("email"),
        "phone": item.get("phone"),
        "organization": item.get("organization"),
        "recurrence_rule": _json_object_or_none(item.get("recurrence_rule")),
        "metadata": _json_object(item.get("metadata")),
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
    }


def list_items(
    user_id: str,
    *,
    item_type: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    parent_id: str | None = None,
    tags: list[str] | None = None,
    query: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    limit = _bounded_limit(limit)
    offset = max(0, offset)
    _validate_optional("type", item_type, ITEM_TYPES)
    _validate_optional("status", status, STATUSES)
    _validate_optional("priority", priority, PRIORITIES)
    rows, has_more = db.list_items(
        user_id,
        item_type=item_type,
        status=status,
        priority=priority,
        parent_id=parent_id,
        tags=tags,
        query=query,
        limit=limit,
        offset=offset,
    )
    return {
        "items": [row_to_item(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
    }


def get_item(user_id: str, item_id: str) -> dict:
    row = db.get_item(item_id, user_id)
    if not row:
        raise ServiceError("item_not_found", "Item not found", status_code=404)
    return row_to_item(row)


def create_item(
    user_id: str,
    payload: dict[str, Any],
    *,
    default_source: str = "manual",
    allow_table: bool = False,
    item_id: str | None = None,
) -> dict:
    data = _prepare_item_payload(
        user_id,
        payload,
        default_source=default_source,
        allow_table=allow_table,
        item_id=item_id or str(uuid.uuid4()),
        existing_row=None,
    )
    db.create_item(data)
    return get_item(user_id, data["id"])


def update_item(user_id: str, item_id: str, payload: dict[str, Any]) -> dict:
    row = db.get_item(item_id, user_id)
    if not row:
        raise ServiceError("item_not_found", "Item not found", status_code=404)
    existing = row_to_item(row)

    if existing["type"] == "decision":
        illegal = set(payload) - PATCHABLE_DECISION_FIELDS
        if illegal:
            raise ServiceError(
                "decision_immutable",
                "Decision title and content are immutable after creation.",
                details={"fields": sorted(illegal)},
            )

    data = _prepare_item_payload(
        user_id,
        payload,
        default_source=existing["source"],
        allow_table=False,
        item_id=item_id,
        existing_row=row,
    )
    updates = {
        key: value
        for key, value in data.items()
        if key not in {"id", "user_id", "created_at"}
    }
    updates["updated_at"] = now_iso()
    db.update_item(item_id, updates)
    return get_item(user_id, item_id)


def delete_item(user_id: str, item_id: str) -> dict:
    item = get_item(user_id, item_id)
    if item["type"] == "table":
        raise ServiceError(
            "table_item_delete_forbidden",
            "Delete the table resource instead of its backing item.",
        )
    db.delete_item(item_id)
    return {"deleted": True}


def create_items_from_capture_results(
    user_id: str,
    capture_content: str,
    results: list[dict[str, Any]],
    *,
    source: str = "voice",
) -> list[dict]:
    if not results:
        results = [{"type": "note", "title": capture_content[:60], "content": capture_content, "tags": []}]

    existing_titles = _title_lookup(user_id)
    planned_titles: dict[str, str] = {}
    planned: list[dict[str, Any]] = []

    for result in results:
        item_id = str(uuid.uuid4())
        payload = dict(result)
        payload.pop("epic_title", None)
        parent_title = payload.pop("parent_title", None) or result.get("epic_title")
        if not payload.get("content"):
            payload["content"] = capture_content
        payload.setdefault("source", source)
        payload.setdefault("tags", [])
        payload["id"] = item_id
        payload["_parent_title"] = parent_title

        title = payload.get("title") or _default_title(payload.get("type", "note"), payload.get("content", ""))
        if title:
            planned_titles[title] = item_id
        planned.append(payload)

    for payload in planned:
        parent_title = payload.pop("_parent_title", None)
        if parent_title and not payload.get("parent_id"):
            payload["parent_id"] = existing_titles.get(parent_title) or planned_titles.get(parent_title)

    pending = {payload["id"]: payload for payload in planned}
    created_ids: set[str] = set()
    while pending:
        progressed = False
        for planned_id, payload in list(pending.items()):
            parent_id = payload.get("parent_id")
            if parent_id and parent_id in pending:
                continue
            create_item(user_id, payload, default_source=source, item_id=planned_id)
            created_ids.add(planned_id)
            pending.pop(planned_id)
            progressed = True
        if not progressed:
            raise ServiceError("parent_cycle", "Capture output contains a parent cycle.")

    return [get_item(user_id, payload["id"]) for payload in planned if payload["id"] in created_ids]


def create_table(user_id: str, payload: dict[str, Any]) -> dict:
    columns = _validate_table_columns(payload.get("columns") or [])
    table_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())
    now = now_iso()
    item = create_item(
        user_id,
        {
            "id": item_id,
            "type": "table",
            "title": payload["title"],
            "content": payload.get("description") or payload["title"],
            "source": "manual",
            "parent_id": payload.get("parent_id"),
            "tags": payload.get("tags") or [],
            "metadata": {"table_id": table_id},
        },
        default_source="manual",
        allow_table=True,
        item_id=item_id,
    )
    record = {
        "id": table_id,
        "user_id": user_id,
        "item_id": item["id"],
        "title": payload["title"],
        "description": payload.get("description"),
        "columns": json.dumps(columns),
        "created_at": now,
        "updated_at": now,
    }
    db.create_table_record(record)
    return table_to_schema(db.get_table(table_id, user_id))


def list_tables(user_id: str, *, limit: int = 100, offset: int = 0) -> dict:
    limit = _bounded_limit(limit)
    offset = max(0, offset)
    rows, has_more = db.list_tables(user_id, limit=limit, offset=offset)
    return {
        "items": [table_to_schema(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
    }


def get_table_detail(user_id: str, table_id: str) -> dict:
    table = _require_table(user_id, table_id)
    rows, _ = db.list_table_rows(table_id, limit=20, offset=0)
    return {
        "table": table_to_schema(table),
        "rows": [table_row_to_schema(row) for row in rows],
    }


def update_table(user_id: str, table_id: str, payload: dict[str, Any]) -> dict:
    table = _require_table(user_id, table_id)
    updates: dict[str, Any] = {}
    if "title" in payload and payload["title"] is not None:
        updates["title"] = payload["title"]
    if "description" in payload:
        updates["description"] = payload["description"]
    if updates:
        updates["updated_at"] = now_iso()
        db.update_table(table_id, updates)
        item_updates = {"updated_at": updates["updated_at"]}
        if "title" in updates:
            item_updates["title"] = updates["title"]
        if "description" in updates:
            item_updates["content"] = updates["description"] or updates.get("title") or table["title"]
        db.update_item(table["item_id"], item_updates)
    return table_to_schema(db.get_table(table_id, user_id))


def delete_table(user_id: str, table_id: str) -> dict:
    table = _require_table(user_id, table_id)
    db.delete_table(table_id, table["item_id"])
    return {"deleted": True}


def create_table_row(user_id: str, table_id: str, payload: dict[str, Any]) -> dict:
    table = _require_table(user_id, table_id)
    data = _validate_table_row_data(table_to_schema(table)["columns"], payload.get("data") or {})
    now = now_iso()
    row = {
        "id": str(uuid.uuid4()),
        "table_id": table_id,
        "data": json.dumps(data),
        "row_order": db.next_table_row_order(table_id),
        "created_at": now,
        "updated_at": now,
    }
    db.create_table_row(row)
    return table_row_to_schema(db.get_table_row(row["id"], table_id))


def list_table_rows(
    user_id: str,
    table_id: str,
    *,
    filter_json: str | None = None,
    order_by: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    table = _require_table(user_id, table_id)
    columns = table_to_schema(table)["columns"]
    column_names = {column["name"] for column in columns}
    filter_data: dict[str, Any] = {}
    if filter_json:
        try:
            parsed = json.loads(filter_json)
        except Exception:
            raise ServiceError("invalid_filter", "filter must be a JSON object string.")
        if not isinstance(parsed, dict):
            raise ServiceError("invalid_filter", "filter must be a JSON object string.")
        unknown = set(parsed) - column_names
        if unknown:
            raise ServiceError("unknown_table_columns", "Filter references unknown columns.", details={"columns": sorted(unknown)})
        filter_data = parsed

    rows, _ = db.list_table_rows(table_id, limit=10_000, offset=0)
    row_schemas = [table_row_to_schema(row) for row in rows]
    if filter_data:
        row_schemas = [
            row for row in row_schemas
            if all(row["data"].get(key) == value for key, value in filter_data.items())
        ]
    if order_by:
        descending = order_by.startswith("-")
        column_name = order_by[1:] if descending else order_by
        if column_name not in column_names:
            raise ServiceError("unknown_table_column", "order_by references an unknown column.")
        row_schemas.sort(key=lambda row: (row["data"].get(column_name) is None, row["data"].get(column_name)), reverse=descending)

    limit = _bounded_limit(limit)
    offset = max(0, offset)
    window = row_schemas[offset: offset + limit + 1]
    return {"items": window[:limit], "limit": limit, "offset": offset, "has_more": len(window) > limit}


def update_table_row(user_id: str, table_id: str, row_id: str, payload: dict[str, Any]) -> dict:
    table = _require_table(user_id, table_id)
    if not db.get_table_row(row_id, table_id):
        raise ServiceError("table_row_not_found", "Table row not found", status_code=404)
    data = _validate_table_row_data(table_to_schema(table)["columns"], payload.get("data") or {})
    db.update_table_row(row_id, {"data": json.dumps(data), "updated_at": now_iso()})
    return table_row_to_schema(db.get_table_row(row_id, table_id))


def delete_table_row(user_id: str, table_id: str, row_id: str) -> dict:
    _require_table(user_id, table_id)
    if not db.get_table_row(row_id, table_id):
        raise ServiceError("table_row_not_found", "Table row not found", status_code=404)
    db.delete_table_row(row_id, table_id)
    return {"deleted": True}


def create_link(user_id: str, payload: dict[str, Any]) -> dict:
    link_type = payload.get("link_type")
    _validate_required("link_type", link_type, LINK_TYPES)
    source_id = payload.get("source_id")
    target_id = payload.get("target_id")
    if source_id == target_id:
        raise ServiceError("invalid_link", "source_id cannot equal target_id.")
    get_item(user_id, source_id)
    get_item(user_id, target_id)
    link = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "source_id": source_id,
        "target_id": target_id,
        "link_type": link_type,
        "created_at": now_iso(),
    }
    try:
        db.create_link(link)
    except sqlite3.IntegrityError:
        raise ServiceError("duplicate_link", "Link already exists.", status_code=409)
    return link_to_schema(db.get_link(link["id"], user_id))


def list_item_links(user_id: str, item_id: str, *, limit: int = 100, offset: int = 0) -> dict:
    get_item(user_id, item_id)
    limit = _bounded_limit(limit)
    offset = max(0, offset)
    rows, has_more = db.list_links_for_item(user_id, item_id, limit=limit, offset=offset)
    return {"items": [link_to_schema(row) for row in rows], "limit": limit, "offset": offset, "has_more": has_more}


def delete_link(user_id: str, link_id: str) -> dict:
    if not db.get_link(link_id, user_id):
        raise ServiceError("link_not_found", "Link not found", status_code=404)
    db.delete_link(link_id, user_id)
    return {"deleted": True}


def create_habit_completion(user_id: str, item_id: str, payload: dict[str, Any]) -> dict:
    item = get_item(user_id, item_id)
    if item["type"] != "habit":
        raise ServiceError("not_a_habit", "Completion endpoints only accept habit items.")
    completed_at = payload.get("completed_at") or now_iso()
    completed_at = _normalize_timestamp(completed_at, "completed_at")
    completed_on = payload.get("completed_on") or completed_at[:10]
    if not ISO_DATE_RE.match(completed_on):
        raise ServiceError("invalid_completed_on", "completed_on must use YYYY-MM-DD.")
    completion = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "item_id": item_id,
        "completed_on": completed_on,
        "completed_at": completed_at,
        "note": payload.get("note"),
        "created_at": now_iso(),
    }
    try:
        db.create_habit_completion(completion)
    except sqlite3.IntegrityError:
        raise ServiceError("duplicate_habit_completion", "Habit completion already exists for that date.", status_code=409)
    return habit_completion_to_schema(db.get_habit_completion(completion["id"], user_id, item_id))


def list_habit_completions(user_id: str, item_id: str, *, limit: int = 100, offset: int = 0) -> dict:
    item = get_item(user_id, item_id)
    if item["type"] != "habit":
        raise ServiceError("not_a_habit", "Completion endpoints only accept habit items.")
    limit = _bounded_limit(limit)
    offset = max(0, offset)
    rows, has_more = db.list_habit_completions(user_id, item_id, limit=limit, offset=offset)
    return {"items": [habit_completion_to_schema(row) for row in rows], "limit": limit, "offset": offset, "has_more": has_more}


def delete_habit_completion(user_id: str, item_id: str, completion_id: str) -> dict:
    if not db.get_habit_completion(completion_id, user_id, item_id):
        raise ServiceError("habit_completion_not_found", "Habit completion not found", status_code=404)
    db.delete_habit_completion(completion_id, user_id, item_id)
    return {"deleted": True}


def create_api_key(user_id: str, label: str | None) -> dict:
    raw_key = "bh_live_" + secrets.token_urlsafe(32)
    now = now_iso()
    record = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "key_hash": hash_api_key(raw_key),
        "label": label,
        "created_at": now,
        "last_used_at": None,
    }
    db.create_api_key(record)
    return {
        "id": record["id"],
        "key": raw_key,
        "label": label,
        "created_at": now,
        "last_used_at": None,
    }


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def list_api_keys(user_id: str) -> list[dict]:
    return [dict(row) for row in db.list_api_keys(user_id)]


def verify_api_key(raw_key: str) -> str | None:
    if not raw_key.startswith("bh_live_"):
        return None
    row = db.get_api_key_by_hash(hash_api_key(raw_key))
    if not row:
        return None
    db.touch_api_key(row["id"])
    return row["user_id"]


def delete_api_key(user_id: str, token_id: str) -> dict:
    db.delete_api_key(token_id, user_id)
    return {"deleted": True}


def get_me(user_id: str) -> dict:
    return {"id": user_id, "item_counts": db.count_items_by_type(user_id)}


def get_tags(user_id: str) -> dict:
    return {"tags": db.list_tags(user_id)}


def create_agent_response(user_id: str, message: str) -> dict:
    from .agent import loop

    return loop.run_agent(user_id, message)


def get_agent_brief(user_id: str) -> dict:
    items = [row_to_item(row) for row in db.list_items_recent(user_id, 50)]
    return {"response": _build_daily_brief(items, "daily brief")}


def lint_items(user_id: str) -> dict:
    rows, _ = db.list_items(user_id, limit=10_000)
    findings = []
    cutoff = datetime.now(UTC) - timedelta(days=30)
    for row in rows:
        item = row_to_item(row)
        if item["type"] == "todo" and item["status"] != "done":
            created_at = _parse_timestamp(item["created_at"])
            if created_at and created_at < cutoff:
                findings.append(
                    {
                        "type": "stale_todo",
                        "item_id": item["id"],
                        "detail": "Open for more than 30 days.",
                    }
                )
        if item["type"] == "habit" and not item["recurrence_rule"]:
            findings.append(
                {
                    "type": "invalid_habit",
                    "item_id": item["id"],
                    "detail": "Habit is missing recurrence_rule.",
                }
            )
    return {
        "findings": findings,
        "summary": f"{len(findings)} finding{'s' if len(findings) != 1 else ''}.",
    }


def table_to_schema(row) -> dict:
    table = dict(row)
    return {
        "id": table["id"],
        "item_id": table["item_id"],
        "title": table["title"],
        "description": table.get("description"),
        "columns": _json_list(table.get("columns")),
        "created_at": table["created_at"],
        "updated_at": table["updated_at"],
    }


def table_row_to_schema(row) -> dict:
    data = dict(row)
    return {
        "id": data["id"],
        "table_id": data["table_id"],
        "data": _json_object(data.get("data")),
        "row_order": data["row_order"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
    }


def link_to_schema(row) -> dict:
    link = dict(row)
    return {
        "id": link["id"],
        "source_id": link["source_id"],
        "target_id": link["target_id"],
        "link_type": link["link_type"],
        "created_at": link["created_at"],
    }


def habit_completion_to_schema(row) -> dict:
    completion = dict(row)
    return {
        "id": completion["id"],
        "item_id": completion["item_id"],
        "completed_on": completion["completed_on"],
        "completed_at": completion["completed_at"],
        "note": completion.get("note"),
        "created_at": completion["created_at"],
    }


def _prepare_item_payload(
    user_id: str,
    payload: dict[str, Any],
    *,
    default_source: str,
    allow_table: bool,
    item_id: str,
    existing_row,
) -> dict:
    existing = row_to_item(existing_row) if existing_row else {}
    data = dict(existing)
    data.update({key: value for key, value in payload.items() if key != "id"})
    data["id"] = item_id
    data["user_id"] = user_id

    used_epic_id = "epic_id" in payload
    if used_epic_id and "parent_id" not in payload:
        data["parent_id"] = payload.get("epic_id")

    item_type = data.get("type") or "note"
    if item_type not in ITEM_TYPES:
        raise ServiceError("invalid_item_type", "Invalid item type.")
    if item_type == "table" and not allow_table:
        raise ServiceError("table_item_create_forbidden", "Table items are created by the table service.")
    if item_type not in PUBLIC_ITEM_TYPES and not allow_table:
        raise ServiceError("invalid_item_type", "Invalid item type.")

    status = data.get("status")
    if "completed" in payload and "status" not in payload:
        status = "done" if payload.get("completed") else "open"
    status = status or "open"
    _validate_required("status", status, STATUSES)

    priority = data.get("priority")
    _validate_optional("priority", priority, PRIORITIES)

    source = data.get("source") or default_source
    _validate_required("source", source, SOURCES)

    title = data.get("title")
    content = data.get("content")
    if item_type == "journal" and not title:
        title = "Journal - " + datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    if not title:
        title = _default_title(item_type, content or "")
    if not content:
        content = title
    if not title:
        raise ServiceError("title_required", "title is required.")

    tags = _normalize_tags(data.get("tags"))
    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ServiceError("invalid_metadata", "metadata must be an object.")

    recurrence_rule = data.get("recurrence_rule")
    if isinstance(recurrence_rule, str):
        recurrence_rule = _json_object_or_none(recurrence_rule)
    if recurrence_rule is not None and not isinstance(recurrence_rule, dict):
        raise ServiceError("invalid_recurrence_rule", "recurrence_rule must be an object.")

    read_status = data.get("read_status")
    if item_type == "resource" and read_status is None:
        read_status = "unread"
    _validate_optional("read_status", read_status, READ_STATUSES)

    normalized = {
        "id": item_id,
        "user_id": user_id,
        "type": item_type,
        "title": title.strip(),
        "content": content,
        "status": status,
        "priority": priority,
        "source": source,
        "parent_id": data.get("parent_id"),
        "tags": json.dumps(tags),
        "due_date": _normalize_optional_timestamp(data.get("due_date"), "due_date"),
        "start_time": _normalize_optional_timestamp(data.get("start_time"), "start_time"),
        "end_time": _normalize_optional_timestamp(data.get("end_time"), "end_time"),
        "location": data.get("location"),
        "url": data.get("url"),
        "read_status": read_status,
        "email": data.get("email"),
        "phone": data.get("phone"),
        "organization": data.get("organization"),
        "recurrence_rule": json.dumps(recurrence_rule) if recurrence_rule is not None else None,
        "metadata": json.dumps(metadata),
        "created_at": existing.get("created_at") or now_iso(),
        "updated_at": existing.get("updated_at") or now_iso(),
        "completed": int(status == "done"),
    }

    _validate_type_specific(normalized, recurrence_rule)
    epic_id = _validate_parent(user_id, normalized, used_epic_id=used_epic_id)
    if epic_id:
        epic = db.get_item(epic_id, user_id)
        if epic:
            tags = _append_tag(tags, epic["title"])
            normalized["tags"] = json.dumps(tags)
    normalized["epic_id"] = epic_id
    return normalized


def _validate_parent(user_id: str, item: dict[str, Any], *, used_epic_id: bool) -> str | None:
    parent_id = item.get("parent_id")
    if item["type"] == "epic" and parent_id:
        raise ServiceError("invalid_parent", "Epic items cannot have a parent in V2.")
    if not parent_id:
        return None
    if parent_id == item["id"]:
        raise ServiceError("invalid_parent", "An item cannot be its own parent.")

    parent_row = db.get_item(parent_id, user_id)
    if not parent_row:
        raise ServiceError("parent_not_found", "Parent item not found.")
    parent = row_to_item(parent_row)
    if used_epic_id and parent["type"] != "epic":
        raise ServiceError("epic_not_found", "Epic not found.")
    if _would_create_parent_cycle(user_id, item["id"], parent_id):
        raise ServiceError("parent_cycle", "Parent cycles are rejected.")
    return parent_id if parent["type"] == "epic" else None


def _would_create_parent_cycle(user_id: str, item_id: str, parent_id: str) -> bool:
    seen = {item_id}
    current = parent_id
    while current:
        if current in seen:
            return True
        seen.add(current)
        row = db.get_item(current, user_id)
        if not row:
            return False
        current = row["parent_id"]
    return False


def _validate_type_specific(item: dict[str, Any], recurrence_rule: dict[str, Any] | None) -> None:
    if item["type"] == "event":
        if not item.get("start_time") or not item.get("end_time"):
            raise ServiceError("event_times_required", "event requires start_time and end_time.")
        start = _parse_timestamp(item["start_time"])
        end = _parse_timestamp(item["end_time"])
        if not start or not end or end <= start:
            raise ServiceError("invalid_event_time", "event end_time must be after start_time.")
    if item["type"] == "habit" and not recurrence_rule:
        raise ServiceError("habit_recurrence_required", "habit requires recurrence_rule.")


def _validate_table_columns(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(columns, list) or not columns:
        raise ServiceError("invalid_table_columns", "columns must be a non-empty array.")
    seen: set[str] = set()
    normalized = []
    for column in columns:
        name = str(column.get("name", "")).strip()
        column_type = column.get("type") or "text"
        if not name:
            raise ServiceError("invalid_table_column", "Column names must be non-empty.")
        if name in seen:
            raise ServiceError("duplicate_table_column", "Column names must be unique.")
        if column_type not in TABLE_COLUMN_TYPES:
            raise ServiceError("invalid_table_column_type", "Invalid table column type.")
        normalized_column = {"name": name, "type": column_type}
        if column_type == "select":
            options = column.get("options")
            if not isinstance(options, list) or not [option for option in options if str(option).strip()]:
                raise ServiceError("invalid_table_select_options", "select columns require non-empty options.")
            normalized_column["options"] = [str(option) for option in options if str(option).strip()]
        elif column.get("options") is not None:
            normalized_column["options"] = column.get("options")
        seen.add(name)
        normalized.append(normalized_column)
    return normalized


def _validate_table_row_data(columns: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ServiceError("invalid_table_row", "Row data must be an object.")
    by_name = {column["name"]: column for column in columns}
    unknown = set(data) - set(by_name)
    if unknown:
        raise ServiceError("unknown_table_columns", "Row data contains unknown columns.", details={"columns": sorted(unknown)})
    for key, value in data.items():
        if value is None:
            continue
        column = by_name[key]
        column_type = column["type"]
        if column_type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ServiceError("invalid_table_value", f"{key} must be numeric.")
        if column_type == "boolean" and not isinstance(value, bool):
            raise ServiceError("invalid_table_value", f"{key} must be boolean.")
        if column_type == "date":
            if not isinstance(value, str) or _parse_timestamp(value) is None:
                raise ServiceError("invalid_table_value", f"{key} must be an ISO 8601 string.")
        if column_type == "select" and value not in set(column.get("options") or []):
            raise ServiceError("invalid_table_value", f"{key} must match a select option.")
        if column_type == "text" and not isinstance(value, str):
            raise ServiceError("invalid_table_value", f"{key} must be text.")
    return data


def _require_table(user_id: str, table_id: str):
    table = db.get_table(table_id, user_id)
    if not table:
        raise ServiceError("table_not_found", "Table not found", status_code=404)
    return table


def _title_lookup(user_id: str) -> dict[str, str]:
    rows, _ = db.list_items(user_id, limit=10_000)
    return {row["title"]: row["id"] for row in rows}


def _build_daily_brief(items: list[dict], message: str) -> str:
    open_todos = [item for item in items if item["type"] == "todo" and item["status"] != "done"]
    events = [item for item in items if item["type"] == "event" and item.get("start_time")]
    if not items:
        return "No items yet. Capture something first, then I can brief or lint it."
    parts = []
    if open_todos:
        parts.append(f"{len(open_todos)} open todo{'s' if len(open_todos) != 1 else ''}")
    if events:
        parts.append(f"{len(events)} scheduled event{'s' if len(events) != 1 else ''}")
    if not parts:
        parts.append(f"{len(items)} recent item{'s' if len(items) != 1 else ''}")
    first = open_todos[0]["title"] if open_todos else items[0]["title"]
    return f"{'; '.join(parts)}. First focus: {first}."


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_object_or_none(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_tags(value: Any) -> list[str]:
    tags = _json_list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        text = str(tag).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _append_tag(tags: list[str], tag: str) -> list[str]:
    normalized_existing = {existing.strip().lower() for existing in tags}
    clean = str(tag).strip()
    if clean and clean.lower() not in normalized_existing:
        return [*tags, clean]
    return tags


def _default_title(item_type: str, content: str) -> str:
    if item_type == "journal":
        return "Journal - " + datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    return (content or item_type).strip()[:60]


def _normalize_optional_timestamp(value: Any, field: str) -> str | None:
    if value in (None, ""):
        return None
    return _normalize_timestamp(value, field)


def _normalize_timestamp(value: Any, field: str) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ServiceError("invalid_timestamp", f"{field} must be an ISO 8601 timestamp.")
    return parsed.astimezone(UTC).isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_required(field: str, value: str | None, allowed: set[str]) -> None:
    if value not in allowed:
        raise ServiceError(f"invalid_{field}", f"{field} must be one of: {', '.join(sorted(allowed))}.")


def _validate_optional(field: str, value: str | None, allowed: set[str]) -> None:
    if value is not None:
        _validate_required(field, value, allowed)


def _bounded_limit(limit: int) -> int:
    return min(max(1, int(limit or 100)), 500)
