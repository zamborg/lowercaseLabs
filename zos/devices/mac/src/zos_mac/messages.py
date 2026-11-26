"""Read-only macOS Messages chat.db incremental FTS5 mirror and search."""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from zos_mac.contacts import ContactsIndex
from zos_mac.typedstream import decode_attributed_body

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
MAX_SEARCH_LIMIT = 50
MAX_TEXT_CHARS = 2000

_FTS_TOKEN_RE = re.compile(r"[^\s\"']+")


def apple_timestamp_to_iso(raw_date: int | float | None) -> str:
    """Convert Apple Messages date (ns or s since 2001-01-01 UTC) to ISO 8601 UTC."""
    if raw_date is None or raw_date == 0:
        return APPLE_EPOCH.isoformat().replace("+00:00", "Z")
    val = float(raw_date)
    seconds = val / 1_000_000_000.0 if abs(val) >= 1e12 else val
    dt = APPLE_EPOCH + timedelta(seconds=seconds)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_fts_query(query: str) -> str:
    """Quote individual tokens so user punctuation never triggers FTS5 syntax errors."""
    tokens = _FTS_TOKEN_RE.findall(query.strip())
    if not tokens:
        return ""
    return " ".join('"' + tok.replace('"', '""') + '"' for tok in tokens)


class MessagesIndex:
    """Local FTS5 index synced incrementally by message.ROWID from read-only chat.db."""

    def __init__(
        self,
        chat_db_path: Path,
        index_db_path: Path,
        *,
        contacts: ContactsIndex | None = None,
    ) -> None:
        self.chat_db_path = chat_db_path.expanduser()
        self.index_db_path = index_db_path.expanduser()
        self.contacts = contacts if contacts is not None else ContactsIndex()

    def _open_index(self) -> sqlite3.Connection:
        self.index_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 2000")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages_store (
                rowid INTEGER PRIMARY KEY,
                handle TEXT NOT NULL,
                is_from_me INTEGER NOT NULL,
                date_iso TEXT NOT NULL,
                chat_identifier TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_store_date_idx ON messages_store (date_iso);
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                text,
                handle,
                chat_identifier,
                content='messages_store',
                content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS messages_store_ai AFTER INSERT ON messages_store BEGIN
                INSERT INTO messages_fts (rowid, text, handle, chat_identifier)
                VALUES (new.rowid, new.text, new.handle, new.chat_identifier);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_store_ad AFTER DELETE ON messages_store BEGIN
                INSERT INTO messages_fts (messages_fts, rowid, text, handle, chat_identifier)
                VALUES ('delete', old.rowid, old.text, old.handle, old.chat_identifier);
            END;
            """
        )
        return conn

    def _open_chat_db_ro(self) -> sqlite3.Connection:
        if not self.chat_db_path.exists():
            raise FileNotFoundError(f"Messages database not found at {self.chat_db_path}")
        uri = f"file:{quote(str(self.chat_db_path), safe='/')}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 2000")
            # Probe readability immediately so macOS TCC/FDA errors surface clearly
            conn.execute("SELECT 1 FROM message LIMIT 1")
            return conn
        except (PermissionError, sqlite3.OperationalError) as exc:
            resolved_py = os.path.realpath(sys.executable)
            raise PermissionError(
                f"Cannot read {self.chat_db_path} ({exc}). "
                f"Grant Full Disk Access in System Settings -> Privacy & Security -> Full Disk Access to: "
                f"{resolved_py}"
            ) from exc

    def sync(self) -> int:
        """Incrementally copy new messages (ROWID > last_rowid) into local FTS5 index."""
        idx_conn = self._open_index()
        try:
            row = idx_conn.execute("SELECT value FROM meta WHERE key = 'last_rowid'").fetchone()
            last_rowid = int(row["value"]) if row else 0

            chat_conn = self._open_chat_db_ro()
            try:
                cur = chat_conn.execute(
                    """
                    SELECT
                        m.ROWID AS rowid,
                        m.text AS text,
                        m.attributedBody AS attributed_body,
                        m.is_from_me AS is_from_me,
                        m.date AS date_raw,
                        COALESCE(h.id, '') AS handle,
                        COALESCE(MIN(c.chat_identifier), '') AS chat_identifier
                    FROM message m
                    LEFT JOIN handle h ON h.ROWID = m.handle_id
                    LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                    LEFT JOIN chat c ON c.ROWID = cmj.chat_id
                    WHERE m.ROWID > ?
                    GROUP BY m.ROWID
                    ORDER BY m.ROWID ASC
                    """,
                    (last_rowid,),
                )
                rows = cur.fetchall()
            finally:
                chat_conn.close()

            if not rows:
                return 0

            inserted = 0
            max_rowid = last_rowid
            with idx_conn:
                for r in rows:
                    rowid = int(r["rowid"])
                    if rowid > max_rowid:
                        max_rowid = rowid
                    raw_text = r["text"]
                    text = raw_text if raw_text else decode_attributed_body(r["attributed_body"])
                    if not text:
                        continue
                    handle = str(r["handle"] or "")
                    chat_id = str(r["chat_identifier"] or "")
                    date_iso = apple_timestamp_to_iso(r["date_raw"])
                    is_from_me = 1 if r["is_from_me"] else 0

                    cur_ins = idx_conn.execute(
                        """
                        INSERT OR IGNORE INTO messages_store
                            (rowid, handle, is_from_me, date_iso, chat_identifier, text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (rowid, handle, is_from_me, date_iso, chat_id, text),
                    )
                    if cur_ins.rowcount:
                        inserted += 1

                idx_conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_rowid', ?)",
                    (str(max_rowid),),
                )
            return inserted
        finally:
            idx_conn.close()

    def search(
        self,
        query: str,
        *,
        handle: str | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Sync new rows from chat.db and run FTS5 search with optional handle/since/limit filters."""
        synced = self.sync()
        clamped_limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        fts_q = _safe_fts_query(query)
        if not fts_q:
            raise ValueError("query must contain at least one search term")

        idx_conn = self._open_index()
        try:
            where_clauses = ["messages_fts MATCH ?"]
            params: list[Any] = [fts_q]

            if handle and handle.strip():
                raw_h = handle.strip()
                like_h = f"%{raw_h}%"
                handle_or: list[str] = ["s.handle LIKE ?", "s.chat_identifier LIKE ?"]
                handle_params: list[Any] = [like_h, like_h]

                resolved_phones, resolved_emails = self.contacts.resolve_handles_for_name(raw_h)
                norm_h_sql = "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(s.handle, '-', ''), ' ', ''), '(', ''), ')', ''), '+', '')"
                norm_c_sql = "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(s.chat_identifier, '-', ''), ' ', ''), '(', ''), ')', ''), '+', '')"
                for phone10 in sorted(resolved_phones):
                    handle_or.append(f"{norm_h_sql} LIKE ?")
                    handle_or.append(f"{norm_c_sql} LIKE ?")
                    handle_params.extend([f"%{phone10}", f"%{phone10}"])
                for email in sorted(resolved_emails):
                    handle_or.append("LOWER(s.handle) = ?")
                    handle_or.append("LOWER(s.chat_identifier) = ?")
                    handle_params.extend([email, email])

                where_clauses.append(f"({' OR '.join(handle_or)})")
                params.extend(handle_params)

            if since and since.strip():
                where_clauses.append("s.date_iso >= ?")
                params.append(since.strip())

            params.append(clamped_limit)
            sql = f"""
                SELECT
                    s.rowid,
                    s.handle,
                    s.is_from_me,
                    s.date_iso,
                    s.chat_identifier,
                    s.text
                FROM messages_fts
                JOIN messages_store s ON s.rowid = messages_fts.rowid
                WHERE {' AND '.join(where_clauses)}
                ORDER BY s.date_iso DESC, s.rowid DESC
                LIMIT ?
            """
            rows = idx_conn.execute(sql, params).fetchall()
        finally:
            idx_conn.close()

        matches = []
        for r in rows:
            txt = str(r["text"])
            truncated = len(txt) > MAX_TEXT_CHARS
            if truncated:
                txt = txt[:MAX_TEXT_CHARS] + "..."
            contact_name = self.contacts.lookup_name(r["handle"]) or self.contacts.lookup_name(
                r["chat_identifier"]
            )
            matches.append(
                {
                    "handle": r["handle"],
                    "contact_name": contact_name,
                    "is_from_me": bool(r["is_from_me"]),
                    "date": r["date_iso"],
                    "chat_identifier": r["chat_identifier"],
                    "text": txt,
                    "truncated": truncated,
                }
            )

        out: dict[str, Any] = {
            "synced_new": synced,
            "count": len(matches),
            "matches": matches,
        }
        if self.contacts.last_hint:
            out["contacts_hint"] = self.contacts.last_hint
        return out
