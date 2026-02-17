from __future__ import annotations

from typing import Any, Optional

from psycopg.types.json import Json

from services.agent.db import Database


def get_agent_job(db: Database, job_id: str) -> Optional[dict]:
    return db.fetch_one("SELECT * FROM agent_jobs WHERE id = %s", (job_id,))


def update_agent_job(db: Database, job_id: str, status: str, output: dict | None, error: str | None) -> None:
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


def get_email(db: Database, email_id: str) -> Optional[dict]:
    return db.fetch_one(
        """
        SELECT id, message_id, subject, date, \"from\", \"to\", cc, bcc, snippet, body_text,
               body_html, labels, is_read, thread_key
        FROM emails
        WHERE id = %s
        """,
        (email_id,),
    )


def get_category(db: Database, category_id: Optional[str]) -> Optional[dict]:
    if not category_id:
        return None
    return db.fetch_one("SELECT * FROM categories WHERE id = %s", (category_id,))


def get_policy(db: Database, policy_id: Optional[str]) -> Optional[dict]:
    if not policy_id:
        return None
    return db.fetch_one(
        "SELECT id, user_id, name, autonomy_level, allowed_actions, reply_template, style_profile, escalation_rules "
        "FROM policies WHERE id = %s",
        (policy_id,),
    )


def list_categories(db: Database, user_id: str) -> list[dict]:
    return db.fetch_all(
        "SELECT id, user_id, name, description, matching_rules, default_policy_id "
        "FROM categories WHERE user_id = %s",
        (user_id,),
    )


def upsert_notes_document(db: Database, user_id: str, title: str) -> dict:
    existing = db.fetch_one(
        "SELECT id, user_id, title, content_markdown, updated_at FROM notes_documents WHERE user_id = %s AND title = %s",
        (user_id, title),
    )
    if existing:
        return existing
    return db.fetch_one(
        "INSERT INTO notes_documents (user_id, title, content_markdown) VALUES (%s, %s, %s) RETURNING *",
        (user_id, title, ""),
    ) or {}


def list_notes_items(db: Database, document_id: str, limit: int = 50) -> list[dict]:
    return db.fetch_all(
        "SELECT id, notes_document_id, email_id, summary, tags, citations, created_at "
        "FROM notes_items WHERE notes_document_id = %s ORDER BY created_at DESC LIMIT %s",
        (document_id, limit),
    )


def update_notes_document_content(db: Database, document_id: str, content: str) -> None:
    db.execute(
        "UPDATE notes_documents SET content_markdown = %s, updated_at = now() WHERE id = %s",
        (content, document_id),
    )


def create_notes_item(
    db: Database,
    document_id: str,
    email_id: str,
    summary: str,
    tags: list[str],
    citations: list[dict[str, Any]],
) -> dict:
    return db.fetch_one(
        """
        INSERT INTO notes_items (notes_document_id, email_id, summary, tags, citations)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING *
        """,
        (document_id, email_id, summary, Json(tags), Json(citations)),
    ) or {}
