"""initial blackhole schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-15 22:15:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


capture_mode = sa.Enum("auto", "note", "todo", "search", name="capturemode")
capture_record_mode = sa.Enum("note", "todo", name="capturerecordmode")
routing_target = sa.Enum("note", "todo", "search", name="routingtarget")
todo_priority = sa.Enum("low", "normal", "high", name="todopriority")
todo_status = sa.Enum("open", "completed", "canceled", name="todostatus")
transcription_status = sa.Enum("created", "uploading", "transcribing", "completed", "failed", name="transcriptionsessionstatus")


def enum_column_type(is_postgres: bool, *values: str, name: str) -> sa.Enum:
    if is_postgres:
        return postgresql.ENUM(*values, name=name, create_type=False)
    return sa.Enum(*values, name=name)


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    capture_mode_col = enum_column_type(is_postgres, "auto", "note", "todo", "search", name="capturemode")
    capture_record_mode_col = enum_column_type(is_postgres, "note", "todo", name="capturerecordmode")
    routing_target_col = enum_column_type(is_postgres, "note", "todo", "search", name="routingtarget")
    todo_priority_col = enum_column_type(is_postgres, "low", "normal", "high", name="todopriority")
    todo_status_col = enum_column_type(is_postgres, "open", "completed", "canceled", name="todostatus")
    transcription_status_col = enum_column_type(
        is_postgres,
        "created",
        "uploading",
        "transcribing",
        "completed",
        "failed",
        name="transcriptionsessionstatus",
    )

    capture_mode.create(bind, checkfirst=True)
    capture_record_mode.create(bind, checkfirst=True)
    routing_target.create(bind, checkfirst=True)
    todo_priority.create(bind, checkfirst=True)
    todo_status.create(bind, checkfirst=True)
    transcription_status.create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("auth_provider", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "audit_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("category", sa.String(length=48), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_messages_category", "audit_messages", ["category"], unique=False)

    op.create_table(
        "blackhole_captures",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("mode", capture_record_mode_col, nullable=False),
        sa.Column("audio_object_key", sa.String(length=512), nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=False),
        sa.Column("transcript_provider", sa.String(length=64), nullable=False),
        sa.Column("routing_target", routing_target_col, nullable=False),
        sa.Column("routing_confidence", sa.Float(), nullable=False),
        sa.Column("routing_payload", sa.JSON(), nullable=False),
        sa.Column("created_record_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_blackhole_captures_created_at", "blackhole_captures", ["created_at"], unique=False)
    op.create_index("ix_blackhole_captures_user_created", "blackhole_captures", ["user_id", "created_at"], unique=False)
    op.create_index("ix_blackhole_captures_user_id", "blackhole_captures", ["user_id"], unique=False)

    op.create_table(
        "blackhole_transcription_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("mode", capture_mode_col, nullable=False),
        sa.Column("status", transcription_status_col, nullable=False),
        sa.Column("audio_object_key", sa.String(length=512), nullable=True),
        sa.Column("final_transcript", sa.Text(), nullable=True),
        sa.Column("route", routing_target_col, nullable=True),
        sa.Column("route_confidence", sa.Float(), nullable=True),
        sa.Column("route_payload", sa.JSON(), nullable=True),
        sa.Column("capture_id", sa.String(length=36), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["capture_id"], ["blackhole_captures.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_blackhole_sessions_user_created",
        "blackhole_transcription_sessions",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_blackhole_transcription_sessions_created_at",
        "blackhole_transcription_sessions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_blackhole_transcription_sessions_status",
        "blackhole_transcription_sessions",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_blackhole_transcription_sessions_user_id",
        "blackhole_transcription_sessions",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "blackhole_notes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("body_markdown", sa.Text(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("source_capture_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_capture_id"], ["blackhole_captures.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_blackhole_notes_archived_at", "blackhole_notes", ["archived_at"], unique=False)
    op.create_index("ix_blackhole_notes_created_at", "blackhole_notes", ["created_at"], unique=False)
    op.create_index("ix_blackhole_notes_updated_at", "blackhole_notes", ["updated_at"], unique=False)
    op.create_index("ix_blackhole_notes_user_id", "blackhole_notes", ["user_id"], unique=False)
    op.create_index(
        "ix_blackhole_notes_user_unarchived_updated",
        "blackhole_notes",
        ["user_id", "archived_at", "updated_at"],
        unique=False,
    )

    op.create_table(
        "blackhole_todos",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description_markdown", sa.Text(), nullable=False),
        sa.Column("priority", todo_priority_col, nullable=False),
        sa.Column("status", todo_status_col, nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("source_capture_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_capture_id"], ["blackhole_captures.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_blackhole_todos_created_at", "blackhole_todos", ["created_at"], unique=False)
    op.create_index("ix_blackhole_todos_status", "blackhole_todos", ["status"], unique=False)
    op.create_index("ix_blackhole_todos_updated_at", "blackhole_todos", ["updated_at"], unique=False)
    op.create_index("ix_blackhole_todos_user_id", "blackhole_todos", ["user_id"], unique=False)
    op.create_index(
        "ix_blackhole_todos_user_status_updated",
        "blackhole_todos",
        ["user_id", "status", "updated_at"],
        unique=False,
    )

    if is_postgres:
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_blackhole_notes_search_vector
            ON blackhole_notes
            USING GIN (
              (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(tags::text, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(body_markdown, '')), 'C')
              )
            )
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    if is_postgres:
        op.execute("DROP INDEX IF EXISTS ix_blackhole_notes_search_vector")

    op.drop_index("ix_blackhole_todos_user_status_updated", table_name="blackhole_todos")
    op.drop_index("ix_blackhole_todos_user_id", table_name="blackhole_todos")
    op.drop_index("ix_blackhole_todos_updated_at", table_name="blackhole_todos")
    op.drop_index("ix_blackhole_todos_status", table_name="blackhole_todos")
    op.drop_index("ix_blackhole_todos_created_at", table_name="blackhole_todos")
    op.drop_table("blackhole_todos")

    op.drop_index("ix_blackhole_notes_user_unarchived_updated", table_name="blackhole_notes")
    op.drop_index("ix_blackhole_notes_user_id", table_name="blackhole_notes")
    op.drop_index("ix_blackhole_notes_updated_at", table_name="blackhole_notes")
    op.drop_index("ix_blackhole_notes_created_at", table_name="blackhole_notes")
    op.drop_index("ix_blackhole_notes_archived_at", table_name="blackhole_notes")
    op.drop_table("blackhole_notes")

    op.drop_index("ix_blackhole_transcription_sessions_user_id", table_name="blackhole_transcription_sessions")
    op.drop_index("ix_blackhole_transcription_sessions_status", table_name="blackhole_transcription_sessions")
    op.drop_index("ix_blackhole_transcription_sessions_created_at", table_name="blackhole_transcription_sessions")
    op.drop_index("ix_blackhole_sessions_user_created", table_name="blackhole_transcription_sessions")
    op.drop_table("blackhole_transcription_sessions")

    op.drop_index("ix_blackhole_captures_user_id", table_name="blackhole_captures")
    op.drop_index("ix_blackhole_captures_user_created", table_name="blackhole_captures")
    op.drop_index("ix_blackhole_captures_created_at", table_name="blackhole_captures")
    op.drop_table("blackhole_captures")

    op.drop_index("ix_audit_messages_category", table_name="audit_messages")
    op.drop_table("audit_messages")
    op.drop_table("users")

    transcription_status.drop(bind, checkfirst=True)
    todo_status.drop(bind, checkfirst=True)
    todo_priority.drop(bind, checkfirst=True)
    routing_target.drop(bind, checkfirst=True)
    capture_record_mode.drop(bind, checkfirst=True)
    capture_mode.drop(bind, checkfirst=True)
