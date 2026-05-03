import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("DB_PATH", "/data/blackhole.db"))
_local = threading.local()

ITEM_COLUMNS = [
    "id",
    "user_id",
    "type",
    "title",
    "content",
    "status",
    "priority",
    "source",
    "parent_id",
    "tags",
    "due_date",
    "start_time",
    "end_time",
    "location",
    "url",
    "read_status",
    "email",
    "phone",
    "organization",
    "recurrence_rule",
    "metadata",
    "created_at",
    "updated_at",
    "epic_id",
    "completed",
]


def get_conn():
    import sqlite3

    if not hasattr(_local, "conn"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def close_conn() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        delattr(_local, "conn")


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            type TEXT NOT NULL DEFAULT 'note',
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            priority TEXT,
            source TEXT NOT NULL DEFAULT 'voice',
            parent_id TEXT REFERENCES items(id),
            tags TEXT NOT NULL DEFAULT '[]',
            due_date TEXT,
            start_time TEXT,
            end_time TEXT,
            location TEXT,
            url TEXT,
            read_status TEXT,
            email TEXT,
            phone TEXT,
            organization TEXT,
            recurrence_rule TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            epic_id TEXT,
            completed INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS llm_logs (
            id TEXT PRIMARY KEY,
            operation TEXT NOT NULL,
            user_id TEXT,
            item_id TEXT,
            model TEXT NOT NULL,
            input_text TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            user_prompt TEXT NOT NULL,
            raw_response TEXT,
            parsed_response TEXT,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tables (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            item_id TEXT NOT NULL UNIQUE REFERENCES items(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            description TEXT,
            columns TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS table_rows (
            id TEXT PRIMARY KEY,
            table_id TEXT NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
            data TEXT NOT NULL DEFAULT '{}',
            row_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS item_links (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            source_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            link_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(source_id, target_id, link_type)
        );

        CREATE TABLE IF NOT EXISTS habit_completions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            completed_on TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, item_id, completed_on)
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            key_hash TEXT NOT NULL UNIQUE,
            label TEXT,
            created_at TEXT NOT NULL,
            last_used_at TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            message TEXT NOT NULL,
            response TEXT,
            context_json TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            tool_turns INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_tool_calls (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(id),
            tool_call_id TEXT,
            name TEXT NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT,
            status TEXT NOT NULL,
            error TEXT,
            duration_ms INTEGER,
            created_at TEXT NOT NULL
        );
    """)
    _migrate_items_table(conn)
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_items_user_id ON items(user_id);
        CREATE INDEX IF NOT EXISTS idx_items_type_status ON items(user_id, type, status);
        CREATE INDEX IF NOT EXISTS idx_items_parent_id ON items(user_id, parent_id);
        CREATE INDEX IF NOT EXISTS idx_items_start_time ON items(user_id, start_time);
        CREATE INDEX IF NOT EXISTS idx_items_due_date ON items(user_id, due_date);
        CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_items_updated_at ON items(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_items_epic_id ON items(epic_id);
        CREATE INDEX IF NOT EXISTS idx_llm_logs_created_at ON llm_logs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_llm_logs_operation_created_at ON llm_logs(operation, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tables_user_id ON tables(user_id);
        CREATE INDEX IF NOT EXISTS idx_table_rows_table_id ON table_rows(table_id, row_order);
        CREATE INDEX IF NOT EXISTS idx_item_links_user_source ON item_links(user_id, source_id);
        CREATE INDEX IF NOT EXISTS idx_item_links_user_target ON item_links(user_id, target_id);
        CREATE INDEX IF NOT EXISTS idx_habit_completions_item ON habit_completions(user_id, item_id, completed_on DESC);
        CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_user_created ON agent_runs(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run ON agent_tool_calls(run_id, created_at);
    """)
    conn.commit()


def _migrate_items_table(conn) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    additions = {
        "status": "TEXT NOT NULL DEFAULT 'open'",
        "priority": "TEXT",
        "source": "TEXT NOT NULL DEFAULT 'voice'",
        "parent_id": "TEXT",
        "start_time": "TEXT",
        "end_time": "TEXT",
        "location": "TEXT",
        "url": "TEXT",
        "read_status": "TEXT",
        "email": "TEXT",
        "phone": "TEXT",
        "organization": "TEXT",
        "recurrence_rule": "TEXT",
        "metadata": "TEXT DEFAULT '{}'",
        "epic_id": "TEXT",
        "completed": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE items ADD COLUMN {name} {definition}")

    conn.execute(
        """
        UPDATE items
        SET status = CASE WHEN COALESCE(completed, 0) = 1 THEN 'done' ELSE 'open' END
        WHERE status IS NULL OR status = ''
        """
    )
    conn.execute("UPDATE items SET source = 'voice' WHERE source IS NULL OR source = ''")
    conn.execute("UPDATE items SET tags = '[]' WHERE tags IS NULL OR tags = ''")
    conn.execute("UPDATE items SET metadata = '{}' WHERE metadata IS NULL OR metadata = ''")
    conn.execute("UPDATE items SET parent_id = epic_id WHERE parent_id IS NULL AND epic_id IS NOT NULL")
    conn.execute("UPDATE items SET completed = CASE WHEN status = 'done' THEN 1 ELSE 0 END")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def upsert_user(user_id: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO users(id, created_at) VALUES (?, ?)",
        (user_id, now_iso()),
    )
    conn.commit()


def create_item(item: dict):
    conn = get_conn()
    values = _item_values(item)
    placeholders = ", ".join(f":{column}" for column in ITEM_COLUMNS)
    columns = ", ".join(ITEM_COLUMNS)
    conn.execute(
        f"INSERT INTO items({columns}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def _item_values(item: dict) -> dict:
    now = now_iso()
    completed = item.get("completed")
    status = item.get("status") or ("done" if completed else "open")
    values = {
        "id": item["id"],
        "user_id": item["user_id"],
        "type": item.get("type", "note"),
        "title": item.get("title") or item.get("content", "")[:60],
        "content": item.get("content") or item.get("title", ""),
        "status": status,
        "priority": item.get("priority"),
        "source": item.get("source") or "voice",
        "parent_id": item.get("parent_id", item.get("epic_id")),
        "tags": item.get("tags") or "[]",
        "due_date": item.get("due_date"),
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
        "location": item.get("location"),
        "url": item.get("url"),
        "read_status": item.get("read_status"),
        "email": item.get("email"),
        "phone": item.get("phone"),
        "organization": item.get("organization"),
        "recurrence_rule": item.get("recurrence_rule"),
        "metadata": item.get("metadata") or "{}",
        "created_at": item.get("created_at") or now,
        "updated_at": item.get("updated_at") or now,
        "epic_id": item.get("epic_id", item.get("parent_id")),
        "completed": int(status == "done"),
    }
    return values


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
) -> tuple[list, bool]:
    conn = get_conn()
    clauses = ["user_id = ?"]
    values: list[Any] = [user_id]
    if item_type:
        clauses.append("type = ?")
        values.append(item_type)
    if status:
        clauses.append("status = ?")
        values.append(status)
    if priority:
        clauses.append("priority = ?")
        values.append(priority)
    if parent_id is not None:
        clauses.append("parent_id = ?")
        values.append(parent_id)

    rows = [
        row
        for row in conn.execute(
            f"""
            SELECT *
            FROM items
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            """,
            values,
        ).fetchall()
    ]

    if tags:
        wanted = {tag.strip().lower() for tag in tags if tag.strip()}
        rows = [row for row in rows if wanted.issubset(_row_tag_set(row))]
    if query:
        q = query.strip().lower()
        rows = [row for row in rows if _row_matches_query(row, q)]

    window = rows[offset: offset + limit + 1]
    return window[:limit], len(window) > limit


def _row_tag_set(row) -> set[str]:
    import json

    try:
        tags = json.loads(row["tags"] or "[]")
    except Exception:
        tags = []
    return {str(tag).strip().lower() for tag in tags}


def _row_matches_query(row, query: str) -> bool:
    if not query:
        return True
    fields = [
        row["title"],
        row["content"],
        row["type"],
        row["status"],
        row["priority"] or "",
        row["tags"] or "",
    ]
    return query in " ".join(str(field).lower() for field in fields)


def list_items_recent(user_id: str, limit: int = 25) -> list:
    rows, _ = list_items(user_id, limit=limit, offset=0)
    return rows


def list_epics(user_id: str) -> list:
    rows, _ = list_items(user_id, item_type="epic", status="open", limit=500, offset=0)
    return rows


def get_item(item_id: str, user_id: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM items WHERE id = ? AND user_id = ?",
        (item_id, user_id),
    ).fetchone()


def get_item_by_title(user_id: str, title: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM items WHERE user_id = ? AND title = ? ORDER BY created_at DESC LIMIT 1",
        (user_id, title),
    ).fetchone()


def update_item(item_id: str, updates: dict):
    if not updates:
        return
    invalid = set(updates) - set(ITEM_COLUMNS)
    if invalid:
        raise ValueError(f"invalid item columns: {', '.join(sorted(invalid))}")
    conn = get_conn()
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [item_id]
    conn.execute(f"UPDATE items SET {set_clause} WHERE id = ?", values)
    conn.commit()


def delete_item(item_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()


def list_all_users() -> list:
    conn = get_conn()
    return conn.execute(
        """
        SELECT
            u.id,
            u.created_at,
            COUNT(i.id) AS item_count,
            COALESCE(SUM(i.completed), 0) AS completed_count,
            MAX(i.updated_at) AS last_activity_at
        FROM users u
        LEFT JOIN items i ON i.user_id = u.id
        GROUP BY u.id, u.created_at
        ORDER BY COALESCE(MAX(i.updated_at), u.created_at) DESC
        """
    ).fetchall()


def list_all_items(limit: int = 500) -> list:
    conn = get_conn()
    return conn.execute(
        """
        SELECT *
        FROM items
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def list_recent_items(limit: int = 20) -> list:
    return list_all_items(limit=limit)


def get_admin_summary() -> dict:
    conn = get_conn()
    summary = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM users) AS total_users,
            (SELECT COUNT(*) FROM items) AS total_items,
            (SELECT COUNT(*) FROM items WHERE status = 'done') AS completed_items,
            (SELECT COUNT(*) FROM items WHERE status != 'done') AS open_items,
            (SELECT COUNT(*) FROM items WHERE due_date IS NOT NULL AND due_date != '') AS items_with_due_dates
        """
    ).fetchone()

    type_rows = conn.execute(
        """
        SELECT type, COUNT(*) AS count
        FROM items
        GROUP BY type
        ORDER BY count DESC, type ASC
        """
    ).fetchall()

    daily_rows = conn.execute(
        """
        SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count
        FROM items
        GROUP BY day
        ORDER BY day DESC
        LIMIT 14
        """
    ).fetchall()

    due_rows = conn.execute(
        """
        SELECT *
        FROM items
        WHERE due_date IS NOT NULL AND due_date != ''
        ORDER BY due_date ASC
        LIMIT 20
        """
    ).fetchall()

    return {
        "total_users": summary["total_users"],
        "total_items": summary["total_items"],
        "completed_items": summary["completed_items"],
        "open_items": summary["open_items"],
        "items_with_due_dates": summary["items_with_due_dates"],
        "types": [dict(row) for row in type_rows],
        "daily_activity": [dict(row) for row in daily_rows],
        "due_items": [dict(row) for row in due_rows],
    }


def create_llm_log(log: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO llm_logs(
            id, operation, user_id, item_id, model, input_text, system_prompt,
            user_prompt, raw_response, parsed_response, status, error, created_at
        ) VALUES (
            :id, :operation, :user_id, :item_id, :model, :input_text, :system_prompt,
            :user_prompt, :raw_response, :parsed_response, :status, :error, :created_at
        )""",
        log,
    )
    conn.commit()


def list_llm_logs(limit: int = 200) -> list:
    conn = get_conn()
    return conn.execute(
        """
        SELECT *
        FROM llm_logs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def get_llm_log_summary() -> dict:
    conn = get_conn()
    summary = conn.execute(
        """
        SELECT
            COUNT(*) AS total_logs,
            COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS failed_logs,
            MAX(created_at) AS last_log_at
        FROM llm_logs
        """
    ).fetchone()

    operation_rows = conn.execute(
        """
        SELECT operation, COUNT(*) AS count
        FROM llm_logs
        GROUP BY operation
        ORDER BY count DESC, operation ASC
        """
    ).fetchall()

    return {
        "total_logs": summary["total_logs"],
        "failed_logs": summary["failed_logs"],
        "last_log_at": summary["last_log_at"],
        "operations": [dict(row) for row in operation_rows],
    }


def count_items_by_type(user_id: str) -> dict[str, int]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT type, COUNT(*) AS count FROM items WHERE user_id = ? GROUP BY type",
        (user_id,),
    ).fetchall()
    return {row["type"]: row["count"] for row in rows}


def list_tags(user_id: str) -> list[str]:
    rows, _ = list_items(user_id, limit=10_000)
    tags: set[str] = set()
    for row in rows:
        tags.update(_row_tag_set(row))
    return sorted(tags)


def create_table_record(table: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO tables(id, user_id, item_id, title, description, columns, created_at, updated_at)
        VALUES (:id, :user_id, :item_id, :title, :description, :columns, :created_at, :updated_at)
        """,
        table,
    )
    conn.commit()


def get_table(table_id: str, user_id: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM tables WHERE id = ? AND user_id = ?",
        (table_id, user_id),
    ).fetchone()


def list_tables(user_id: str, *, limit: int = 100, offset: int = 0) -> tuple[list, bool]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT *
        FROM tables
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (user_id, limit + 1, offset),
    ).fetchall()
    return rows[:limit], len(rows) > limit


def update_table(table_id: str, updates: dict):
    if not updates:
        return
    invalid = set(updates) - {"title", "description", "updated_at"}
    if invalid:
        raise ValueError(f"invalid table columns: {', '.join(sorted(invalid))}")
    conn = get_conn()
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [table_id]
    conn.execute(f"UPDATE tables SET {set_clause} WHERE id = ?", values)
    conn.commit()


def delete_table(table_id: str, item_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM tables WHERE id = ?", (table_id,))
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()


def create_table_row(row: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO table_rows(id, table_id, data, row_order, created_at, updated_at)
        VALUES (:id, :table_id, :data, :row_order, :created_at, :updated_at)
        """,
        row,
    )
    conn.commit()


def get_table_row(row_id: str, table_id: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM table_rows WHERE id = ? AND table_id = ?",
        (row_id, table_id),
    ).fetchone()


def list_table_rows(table_id: str, *, limit: int = 100, offset: int = 0) -> tuple[list, bool]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT *
        FROM table_rows
        WHERE table_id = ?
        ORDER BY row_order ASC, created_at ASC
        LIMIT ? OFFSET ?
        """,
        (table_id, limit + 1, offset),
    ).fetchall()
    return rows[:limit], len(rows) > limit


def next_table_row_order(table_id: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COALESCE(MAX(row_order), -1) + 1 AS next_order FROM table_rows WHERE table_id = ?",
        (table_id,),
    ).fetchone()
    return int(row["next_order"])


def update_table_row(row_id: str, updates: dict):
    if not updates:
        return
    invalid = set(updates) - {"data", "updated_at"}
    if invalid:
        raise ValueError(f"invalid table row columns: {', '.join(sorted(invalid))}")
    conn = get_conn()
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [row_id]
    conn.execute(f"UPDATE table_rows SET {set_clause} WHERE id = ?", values)
    conn.commit()


def delete_table_row(row_id: str, table_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM table_rows WHERE id = ? AND table_id = ?", (row_id, table_id))
    conn.commit()


def create_link(link: dict):
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO item_links(id, user_id, source_id, target_id, link_type, created_at)
            VALUES (:id, :user_id, :source_id, :target_id, :link_type, :created_at)
            """,
            link,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_link(link_id: str, user_id: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM item_links WHERE id = ? AND user_id = ?",
        (link_id, user_id),
    ).fetchone()


def list_links_for_item(user_id: str, item_id: str, *, limit: int = 100, offset: int = 0) -> tuple[list, bool]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT *
        FROM item_links
        WHERE user_id = ? AND (source_id = ? OR target_id = ?)
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (user_id, item_id, item_id, limit + 1, offset),
    ).fetchall()
    return rows[:limit], len(rows) > limit


def delete_link(link_id: str, user_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM item_links WHERE id = ? AND user_id = ?", (link_id, user_id))
    conn.commit()


def create_habit_completion(completion: dict):
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO habit_completions(
                id, user_id, item_id, completed_on, completed_at, note, created_at
            ) VALUES (
                :id, :user_id, :item_id, :completed_on, :completed_at, :note, :created_at
            )
            """,
            completion,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_habit_completion(completion_id: str, user_id: str, item_id: str):
    conn = get_conn()
    return conn.execute(
        """
        SELECT *
        FROM habit_completions
        WHERE id = ? AND user_id = ? AND item_id = ?
        """,
        (completion_id, user_id, item_id),
    ).fetchone()


def list_habit_completions(user_id: str, item_id: str, *, limit: int = 100, offset: int = 0) -> tuple[list, bool]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT *
        FROM habit_completions
        WHERE user_id = ? AND item_id = ?
        ORDER BY completed_on DESC
        LIMIT ? OFFSET ?
        """,
        (user_id, item_id, limit + 1, offset),
    ).fetchall()
    return rows[:limit], len(rows) > limit


def delete_habit_completion(completion_id: str, user_id: str, item_id: str):
    conn = get_conn()
    conn.execute(
        "DELETE FROM habit_completions WHERE id = ? AND user_id = ? AND item_id = ?",
        (completion_id, user_id, item_id),
    )
    conn.commit()


def create_api_key(record: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO api_keys(id, user_id, key_hash, label, created_at, last_used_at)
        VALUES (:id, :user_id, :key_hash, :label, :created_at, :last_used_at)
        """,
        record,
    )
    conn.commit()


def list_api_keys(user_id: str) -> list:
    conn = get_conn()
    return conn.execute(
        """
        SELECT id, label, created_at, last_used_at
        FROM api_keys
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()


def get_api_key_by_hash(key_hash: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM api_keys WHERE key_hash = ?",
        (key_hash,),
    ).fetchone()


def touch_api_key(key_id: str):
    conn = get_conn()
    conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now_iso(), key_id))
    conn.commit()


def delete_api_key(token_id: str, user_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM api_keys WHERE id = ? AND user_id = ?", (token_id, user_id))
    conn.commit()


def create_agent_run(run: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO agent_runs(
            id, user_id, message, response, context_json, model, status, error,
            tool_turns, created_at, completed_at
        ) VALUES (
            :id, :user_id, :message, :response, :context_json, :model, :status, :error,
            :tool_turns, :created_at, :completed_at
        )
        """,
        run,
    )
    conn.commit()


def update_agent_run(run_id: str, updates: dict):
    if not updates:
        return
    invalid = set(updates) - {"response", "status", "error", "tool_turns", "completed_at"}
    if invalid:
        raise ValueError(f"invalid agent run columns: {', '.join(sorted(invalid))}")
    conn = get_conn()
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [run_id]
    conn.execute(f"UPDATE agent_runs SET {set_clause} WHERE id = ?", values)
    conn.commit()


def create_agent_tool_call(call: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO agent_tool_calls(
            id, run_id, user_id, tool_call_id, name, input_json, output_json,
            status, error, duration_ms, created_at
        ) VALUES (
            :id, :run_id, :user_id, :tool_call_id, :name, :input_json, :output_json,
            :status, :error, :duration_ms, :created_at
        )
        """,
        call,
    )
    conn.commit()
