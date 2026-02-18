"""Initial schema for theVoid backend.

Revision ID: 0001_initial
Revises:
Create Date: 2026-02-18 00:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


ENTRY_STATUS_ENUM = sa.Enum(
    "UPLOADING",
    "UPLOADED",
    "TRANSCRIBING",
    "ANALYZING",
    "READY",
    "FAILED",
    name="entrystatus",
)

REVEAL_MODE_ENUM = sa.Enum(
    "ANONYMOUS",
    "REVEALED_TO_FRIENDS",
    "REVEALED_TO_SPECIFIC",
    name="revealmode",
)

JOB_TYPE_ENUM = sa.Enum(
    "TRANSCRIPTION",
    "INSIGHTS",
    "SOCIAL_AGGREGATION",
    name="jobtype",
)

JOB_STATUS_ENUM = sa.Enum(
    "PENDING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    name="jobstatus",
)


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(index_name == index["name"] for index in inspector.get_indexes(table_name))


def _create_index_if_missing(
    index_name: str,
    table_name: str,
    columns: list[str],
    unique: bool = False,
) -> None:
    if not _table_exists(table_name):
        return
    if _index_exists(table_name, index_name):
        return
    op.create_index(index_name, table_name, columns, unique=unique)


def upgrade() -> None:
    if not _table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("apple_sub", sa.String(length=255), nullable=False),
            sa.Column("display_name", sa.String(length=120), nullable=True),
            sa.Column("anonymous_handle", sa.String(length=80), nullable=False),
            sa.Column("daily_checkin_time_local", sa.String(length=8), nullable=False),
            sa.Column("timezone", sa.String(length=64), nullable=False),
            sa.Column("notification_enabled", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("entries"):
        op.create_table(
            "entries",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("local_date", sa.Date(), nullable=False),
            sa.Column("audio_object_key", sa.String(length=512), nullable=False),
            sa.Column("duration_seconds", sa.Integer(), nullable=False),
            sa.Column("status", ENTRY_STATUS_ENUM, nullable=False),
            sa.Column("trace_id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("audio_object_key"),
        )

    if not _table_exists("transcripts"):
        op.create_table(
            "transcripts",
            sa.Column("entry_id", sa.String(length=36), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("provider_metadata", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["entry_id"], ["entries.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("entry_id"),
        )

    if not _table_exists("insights"):
        op.create_table(
            "insights",
            sa.Column("entry_id", sa.String(length=36), nullable=False),
            sa.Column("mood_score", sa.Float(), nullable=False),
            sa.Column("mood_tags", sa.JSON(), nullable=False),
            sa.Column("summary", sa.Text(), nullable=False),
            sa.Column("themes", sa.JSON(), nullable=False),
            sa.Column("signals", sa.JSON(), nullable=False),
            sa.Column("safety_flags", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["entry_id"], ["entries.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("entry_id"),
        )

    if not _table_exists("friend_edges"):
        op.create_table(
            "friend_edges",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("friend_user_id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["friend_user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "friend_user_id", name="uq_friend_edge"),
        )

    if not _table_exists("social_presence"):
        op.create_table(
            "social_presence",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("local_date", sa.Date(), nullable=False),
            sa.Column("dot_color", sa.String(length=32), nullable=False),
            sa.Column("reveal_mode", REVEAL_MODE_ENUM, nullable=False),
            sa.Column("reveal_friend_ids", sa.JSON(), nullable=False),
            sa.Column("display_name_override", sa.String(length=120), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "local_date", name="uq_social_presence_user_day"),
        )

    if not _table_exists("invite_tokens"):
        op.create_table(
            "invite_tokens",
            sa.Column("token", sa.String(length=128), nullable=False),
            sa.Column("inviter_user_id", sa.String(length=36), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("max_uses", sa.Integer(), nullable=False),
            sa.Column("use_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["inviter_user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("token"),
        )

    if not _table_exists("jobs"):
        op.create_table(
            "jobs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("job_type", JOB_TYPE_ENUM, nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("status", JOB_STATUS_ENUM, nullable=False),
            sa.Column("attempt_count", sa.Integer(), nullable=False),
            sa.Column("max_attempts", sa.Integer(), nullable=False),
            sa.Column("trace_id", sa.String(length=36), nullable=False),
            sa.Column("run_after", sa.DateTime(timezone=True), nullable=False),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    # Users
    _create_index_if_missing("ix_users_apple_sub", "users", ["apple_sub"], unique=True)
    _create_index_if_missing("ix_users_anonymous_handle", "users", ["anonymous_handle"], unique=True)

    # Entries
    _create_index_if_missing("ix_entries_user_id", "entries", ["user_id"])
    _create_index_if_missing("ix_entries_local_date", "entries", ["local_date"])
    _create_index_if_missing("ix_entries_status", "entries", ["status"])
    _create_index_if_missing("ix_entries_trace_id", "entries", ["trace_id"])

    # Friend edges
    _create_index_if_missing("ix_friend_edges_user_id", "friend_edges", ["user_id"])
    _create_index_if_missing("ix_friend_edges_friend_user_id", "friend_edges", ["friend_user_id"])

    # Social presence
    _create_index_if_missing("ix_social_presence_user_id", "social_presence", ["user_id"])
    _create_index_if_missing("ix_social_presence_local_date", "social_presence", ["local_date"])
    _create_index_if_missing("ix_social_presence_reveal_mode", "social_presence", ["reveal_mode"])

    # Invite tokens
    _create_index_if_missing("ix_invite_tokens_inviter_user_id", "invite_tokens", ["inviter_user_id"])
    _create_index_if_missing("ix_invite_tokens_expires_at", "invite_tokens", ["expires_at"])

    # Jobs
    _create_index_if_missing("ix_jobs_job_type", "jobs", ["job_type"])
    _create_index_if_missing("ix_jobs_status", "jobs", ["status"])
    _create_index_if_missing("ix_jobs_trace_id", "jobs", ["trace_id"])
    _create_index_if_missing("ix_jobs_run_after", "jobs", ["run_after"])


def downgrade() -> None:
    if _table_exists("jobs"):
        op.drop_table("jobs")
    if _table_exists("invite_tokens"):
        op.drop_table("invite_tokens")
    if _table_exists("social_presence"):
        op.drop_table("social_presence")
    if _table_exists("friend_edges"):
        op.drop_table("friend_edges")
    if _table_exists("insights"):
        op.drop_table("insights")
    if _table_exists("transcripts"):
        op.drop_table("transcripts")
    if _table_exists("entries"):
        op.drop_table("entries")
    if _table_exists("users"):
        op.drop_table("users")

    if op.get_bind().dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS jobstatus"))
        op.execute(sa.text("DROP TYPE IF EXISTS jobtype"))
        op.execute(sa.text("DROP TYPE IF EXISTS revealmode"))
        op.execute(sa.text("DROP TYPE IF EXISTS entrystatus"))
