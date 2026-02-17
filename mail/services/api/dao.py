from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from psycopg.types.json import Json

from services.api.db import Database


def ensure_user(db: Database, email_address: str) -> dict:
    row = db.fetch_one("SELECT * FROM users WHERE email_address = %s", (email_address,))
    if row:
        return row
    return db.fetch_one(
        "INSERT INTO users (email_address) VALUES (%s) RETURNING *",
        (email_address,),
    ) or {}


def ensure_account(db: Database, user_id: str, account_email: str) -> dict:
    row = db.fetch_one("SELECT * FROM accounts WHERE user_id = %s", (user_id,))
    if row:
        return row
    return db.fetch_one(
        """
        INSERT INTO accounts (user_id, provider, imap_user, smtp_user)
        VALUES (%s, %s, %s, %s)
        RETURNING *
        """,
        (user_id, "imap", account_email, account_email),
    ) or {}


def list_emails(db: Database, account_id: str, limit: int, view: str) -> list[dict]:
    clause = "" if view == "all" else "AND is_read = false"
    query = (
        "SELECT id::text AS id, subject, snippet, date::text AS date, \"from\", \"to\", is_read, thread_key "
        "FROM emails WHERE account_id = %s "
        + clause
        + " ORDER BY date DESC NULLS LAST LIMIT %s"
    )
    return db.fetch_all(query, (account_id, limit))


def get_email(db: Database, email_id: str) -> Optional[dict]:
    return db.fetch_one(
        """
        SELECT id::text AS id, message_id, subject, date::text AS date, \"from\", \"to\", cc, bcc, snippet, body_text, body_html,
               labels, is_read, thread_key
        FROM emails
        WHERE id = %s
        """,
        (email_id,),
    )


def get_thread_context(db: Database, thread_key: str, before_date: Optional[str], limit: int) -> list[dict]:
    if not thread_key:
        return []
    params: list[Any] = [thread_key]
    date_clause = ""
    if before_date:
        date_clause = "AND date < %s"
        params.append(before_date)
    params.append(limit)
    query = (
        "SELECT id::text AS id, subject, date::text AS date, \"from\", snippet FROM emails "
        "WHERE thread_key = %s "
        + date_clause
        + " ORDER BY date DESC NULLS LAST LIMIT %s"
    )
    return db.fetch_all(query, params)


def insert_triage_event(
    db: Database,
    user_id: str,
    email_id: str,
    action: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    db.execute(
        "INSERT INTO triage_events (user_id, email_id, action, payload) VALUES (%s, %s, %s, %s)",
        (user_id, email_id, action, Json(payload or {})),
    )


def get_latest_triage_events(db: Database, user_id: str, email_ids: Iterable[str]) -> dict:
    email_ids = list(email_ids)
    if not email_ids:
        return {}
    query = (
        "SELECT DISTINCT ON (email_id) email_id, action, payload, created_at "
        "FROM triage_events WHERE user_id = %s AND email_id = ANY(%s) "
        "ORDER BY email_id, created_at DESC"
    )
    rows = db.fetch_all(query, (user_id, email_ids))
    return {row["email_id"]: row for row in rows}


def list_categories(db: Database, user_id: str) -> list[dict]:
    return db.fetch_all(
        "SELECT id::text AS id, user_id::text AS user_id, name, description, matching_rules, "
        "default_policy_id::text AS default_policy_id "
        "FROM categories WHERE user_id = %s ORDER BY name",
        (user_id,),
    )


def create_category(db: Database, user_id: str, payload: dict) -> dict:
    return db.fetch_one(
        """
        INSERT INTO categories (user_id, name, description, matching_rules, default_policy_id)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id::text AS id, user_id::text AS user_id, name, description, matching_rules,
                  default_policy_id::text AS default_policy_id
        """,
        (
            user_id,
            payload.get("name"),
            payload.get("description"),
            Json(payload.get("matching_rules") or {}),
            payload.get("default_policy_id"),
        ),
    ) or {}


def update_category(db: Database, category_id: str, user_id: str, payload: dict) -> Optional[dict]:
    db.execute(
        """
        UPDATE categories
        SET name = %s,
            description = %s,
            matching_rules = %s,
            default_policy_id = %s
        WHERE id = %s AND user_id = %s
        """,
        (
            payload.get("name"),
            payload.get("description"),
            Json(payload.get("matching_rules") or {}),
            payload.get("default_policy_id"),
            category_id,
            user_id,
        ),
    )
    return db.fetch_one(
        "SELECT id::text AS id, user_id::text AS user_id, name, description, matching_rules, "
        "default_policy_id::text AS default_policy_id "
        "FROM categories WHERE id = %s AND user_id = %s",
        (category_id, user_id),
    )


def list_policies(db: Database, user_id: str) -> list[dict]:
    return db.fetch_all(
        "SELECT id::text AS id, user_id::text AS user_id, name, autonomy_level, allowed_actions, "
        "reply_template, style_profile, escalation_rules FROM policies WHERE user_id = %s ORDER BY name",
        (user_id,),
    )


def create_policy(db: Database, user_id: str, payload: dict) -> dict:
    return db.fetch_one(
        """
        INSERT INTO policies (user_id, name, autonomy_level, allowed_actions, reply_template,
                             style_profile, escalation_rules)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id::text AS id, user_id::text AS user_id, name, autonomy_level, allowed_actions,
                  reply_template, style_profile, escalation_rules
        """,
        (
            user_id,
            payload.get("name"),
            payload.get("autonomy_level"),
            Json(payload.get("allowed_actions") or []),
            payload.get("reply_template"),
            Json(payload.get("style_profile") or {}),
            Json(payload.get("escalation_rules") or {}),
        ),
    ) or {}


def update_policy(db: Database, policy_id: str, user_id: str, payload: dict) -> Optional[dict]:
    db.execute(
        """
        UPDATE policies
        SET name = %s,
            autonomy_level = %s,
            allowed_actions = %s,
            reply_template = %s,
            style_profile = %s,
            escalation_rules = %s
        WHERE id = %s AND user_id = %s
        """,
        (
            payload.get("name"),
            payload.get("autonomy_level"),
            Json(payload.get("allowed_actions") or []),
            payload.get("reply_template"),
            Json(payload.get("style_profile") or {}),
            Json(payload.get("escalation_rules") or {}),
            policy_id,
            user_id,
        ),
    )
    return db.fetch_one(
        "SELECT id::text AS id, user_id::text AS user_id, name, autonomy_level, allowed_actions, "
        "reply_template, style_profile, escalation_rules FROM policies WHERE id = %s AND user_id = %s",
        (policy_id, user_id),
    )


def get_policy(db: Database, policy_id: Optional[str]) -> Optional[dict]:
    if not policy_id:
        return None
    return db.fetch_one(
        "SELECT id::text AS id, user_id::text AS user_id, name, autonomy_level, allowed_actions, "
        "reply_template, style_profile, escalation_rules FROM policies WHERE id = %s",
        (policy_id,),
    )


def create_agent_job(db: Database, user_id: str, job_type: str, input_refs: dict) -> dict:
    return db.fetch_one(
        "INSERT INTO agent_jobs (user_id, type, status, input_refs) VALUES (%s, %s, %s, %s) RETURNING *",
        (user_id, job_type, "queued", Json(input_refs)),
    ) or {}


def get_agent_job(db: Database, job_id: str) -> Optional[dict]:
    return db.fetch_one("SELECT * FROM agent_jobs WHERE id = %s", (job_id,))


def update_agent_job(db: Database, job_id: str, status: str, output: dict | None = None, error: str | None = None) -> None:
    db.execute(
        """
        UPDATE agent_jobs
        SET status = %s,
            output = %s,
            error = %s,
            updated_at = now()
        WHERE id = %s
        """,
        (status, Json(output or {}), error, job_id),
    )


def get_agent_hint(db: Database, user_id: str, email_id: str) -> Optional[dict]:
    email_id_text = str(email_id)
    return db.fetch_one(
        """
        SELECT id::text AS id, type, status, output
        FROM agent_jobs
        WHERE user_id = %s AND input_refs->>'email_id' = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id, email_id_text),
    )


def list_notes_documents(db: Database, user_id: str) -> list[dict]:
    return db.fetch_all(
        "SELECT id::text AS id, user_id::text AS user_id, title, content_markdown, "
        "updated_at::text AS updated_at FROM notes_documents WHERE user_id = %s ORDER BY updated_at DESC",
        (user_id,),
    )


def get_notes_document(db: Database, doc_id: str, user_id: str) -> Optional[dict]:
    return db.fetch_one(
        "SELECT id::text AS id, user_id::text AS user_id, title, content_markdown, "
        "updated_at::text AS updated_at FROM notes_documents WHERE id = %s AND user_id = %s",
        (doc_id, user_id),
    )


def upsert_notes_document(db: Database, user_id: str, title: str) -> dict:
    existing = db.fetch_one(
        "SELECT id::text AS id, user_id::text AS user_id, title, content_markdown, "
        "updated_at::text AS updated_at FROM notes_documents WHERE user_id = %s AND title = %s",
        (user_id, title),
    )
    if existing:
        return existing
    db.execute(
        "INSERT INTO notes_documents (user_id, title, content_markdown) VALUES (%s, %s, %s)",
        (user_id, title, ""),
    )
    return db.fetch_one(
        "SELECT id::text AS id, user_id::text AS user_id, title, content_markdown, "
        "updated_at::text AS updated_at FROM notes_documents WHERE user_id = %s AND title = %s",
        (user_id, title),
    ) or {}


def append_notes_item(
    db: Database,
    document_id: str,
    email_id: str,
    summary: str,
    tags: list[str],
    citations: list[dict],
) -> dict:
    db.execute(
        """
        INSERT INTO notes_items (notes_document_id, email_id, summary, tags, citations)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (document_id, email_id, summary, Json(tags), Json(citations)),
    )
    db.execute(
        "UPDATE notes_documents SET updated_at = now() WHERE id = %s",
        (document_id,),
    )
    return db.fetch_one(
        "SELECT id, notes_document_id, email_id, summary, tags, citations, created_at FROM notes_items WHERE notes_document_id = %s ORDER BY created_at DESC LIMIT 1",
        (document_id,),
    ) or {}
