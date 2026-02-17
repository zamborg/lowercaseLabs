from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool


class Database:
    def __init__(self, database_url: str) -> None:
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
        )
        self._schema_path = Path(__file__).resolve().parents[1] / "api" / "schema.sql"

    def close(self) -> None:
        self._pool.close()

    def init_schema(self) -> None:
        schema_sql = self._schema_path.read_text(encoding="utf-8")
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)

    def fetch_one(self, query: str, params: Optional[Iterable[Any]] = None) -> Optional[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.fetchone()

    def execute(self, query: str, params: Optional[Iterable[Any]] = None) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                conn.commit()

    def ensure_user(self, email_address: str) -> dict:
        row = self.fetch_one("SELECT * FROM users WHERE email_address = %s", (email_address,))
        if row:
            return row
        return self.fetch_one(
            "INSERT INTO users (email_address) VALUES (%s) RETURNING *",
            (email_address,),
        ) or {}

    def ensure_account(
        self,
        user_id: str,
        account_email: str,
        imap_host: Optional[str] = None,
        imap_user: Optional[str] = None,
        imap_encrypted_pass: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_user: Optional[str] = None,
        smtp_encrypted_pass: Optional[str] = None,
    ) -> dict:
        row = self.fetch_one("SELECT * FROM accounts WHERE user_id = %s", (user_id,))
        if row:
            return row
        return self.fetch_one(
            """
            INSERT INTO accounts (user_id, provider, imap_host, imap_user, imap_encrypted_pass,
                                  smtp_host, smtp_user, smtp_encrypted_pass)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                user_id,
                "imap",
                imap_host,
                imap_user,
                imap_encrypted_pass,
                smtp_host,
                smtp_user,
                smtp_encrypted_pass,
            ),
        ) or {}

    def email_exists(self, account_id: str, message_id: str) -> bool:
        row = self.fetch_one(
            "SELECT id FROM emails WHERE account_id = %s AND message_id = %s",
            (account_id, message_id),
        )
        return bool(row)

    def insert_email(self, account_id: str, payload: dict) -> None:
        self.execute(
            """
            INSERT INTO emails (account_id, message_id, thread_key, "from", "to", cc, bcc,
                                subject, date, snippet, body_text, body_html, labels, is_read)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                account_id,
                payload.get("message_id"),
                payload.get("thread_key"),
                Json(payload.get("from") or []),
                Json(payload.get("to") or []),
                Json(payload.get("cc") or []),
                Json(payload.get("bcc") or []),
                payload.get("subject"),
                payload.get("date"),
                payload.get("snippet"),
                payload.get("body_text"),
                payload.get("body_html"),
                Json(payload.get("labels") or {}),
                bool(payload.get("is_read", False)),
            ),
        )
