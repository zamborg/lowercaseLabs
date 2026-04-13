import os
import threading
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "/data/blackhole.db"))
_local = threading.local()


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
            content TEXT NOT NULL,
            title TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'note',
            due_date TEXT,
            completed INTEGER NOT NULL DEFAULT 0,
            tags TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
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

        CREATE INDEX IF NOT EXISTS idx_items_user_id ON items(user_id);
        CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_llm_logs_created_at ON llm_logs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_llm_logs_operation_created_at ON llm_logs(operation, created_at DESC);
    """)
    conn.commit()


def upsert_user(user_id: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO users(id, created_at) VALUES (?, ?)",
        (user_id, datetime.utcnow().isoformat()),
    )
    conn.commit()


def create_item(item: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO items(id, user_id, content, title, type, due_date, completed, tags, created_at, updated_at)
           VALUES (:id, :user_id, :content, :title, :type, :due_date, :completed, :tags, :created_at, :updated_at)""",
        item,
    )
    conn.commit()


def list_items(user_id: str) -> list:
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM items WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()


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
            (SELECT COUNT(*) FROM items WHERE completed = 1) AS completed_items,
            (SELECT COUNT(*) FROM items WHERE completed = 0) AS open_items,
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


def get_item(item_id: str, user_id: str):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM items WHERE id = ? AND user_id = ?",
        (item_id, user_id),
    ).fetchone()


def update_item(item_id: str, updates: dict):
    conn = get_conn()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [item_id]
    conn.execute(f"UPDATE items SET {set_clause} WHERE id = ?", values)
    conn.commit()


def delete_item(item_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
