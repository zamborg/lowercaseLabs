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

        CREATE INDEX IF NOT EXISTS idx_items_user_id ON items(user_id);
        CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(created_at DESC);
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
