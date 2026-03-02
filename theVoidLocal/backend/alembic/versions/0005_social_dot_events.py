"""Add append-only social dot events table.

Revision ID: 0005_social_dot_events
Revises: 0004_feedback_reports
Create Date: 2026-03-02 00:10:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0005_social_dot_events"
down_revision = "0004_feedback_reports"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return False
    return any(index_name == index["name"] for index in inspector.get_indexes(table_name))


def _column_exists(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    json_default = sa.text("'[]'::json") if dialect == "postgresql" else sa.text("'[]'")

    if not _table_exists("social_dot_events"):
        op.create_table(
            "social_dot_events",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("local_date", sa.Date(), nullable=False),
            sa.Column("dot_color", sa.String(length=32), nullable=False),
            sa.Column("dot_tags", sa.JSON(), nullable=False, server_default=json_default),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.alter_column("social_dot_events", "dot_tags", server_default=None)

    if not _index_exists("social_dot_events", "ix_social_dot_events_user_id"):
        op.create_index("ix_social_dot_events_user_id", "social_dot_events", ["user_id"], unique=False)
    if not _index_exists("social_dot_events", "ix_social_dot_events_local_date"):
        op.create_index("ix_social_dot_events_local_date", "social_dot_events", ["local_date"], unique=False)
    if not _index_exists("social_dot_events", "ix_social_dot_events_updated_at"):
        op.create_index("ix_social_dot_events_updated_at", "social_dot_events", ["updated_at"], unique=False)

    if _table_exists("social_presence"):
        has_dot_tags = _column_exists("social_presence", "dot_tags")
        if has_dot_tags:
            dot_tags_sql = "COALESCE(sp.dot_tags, '[]'::json)" if dialect == "postgresql" else "COALESCE(sp.dot_tags, '[]')"
        else:
            dot_tags_sql = "'[]'::json" if dialect == "postgresql" else "'[]'"

        op.execute(
            sa.text(
                f"""
                INSERT INTO social_dot_events (id, user_id, local_date, dot_color, dot_tags, created_at, updated_at)
                SELECT sp.id, sp.user_id, sp.local_date, sp.dot_color, {dot_tags_sql}, sp.created_at, sp.updated_at
                FROM social_presence sp
                LEFT JOIN social_dot_events sde ON sde.id = sp.id
                WHERE sde.id IS NULL
                """
            )
        )


def downgrade() -> None:
    if not _table_exists("social_dot_events"):
        return

    if _index_exists("social_dot_events", "ix_social_dot_events_updated_at"):
        op.drop_index("ix_social_dot_events_updated_at", table_name="social_dot_events")
    if _index_exists("social_dot_events", "ix_social_dot_events_local_date"):
        op.drop_index("ix_social_dot_events_local_date", table_name="social_dot_events")
    if _index_exists("social_dot_events", "ix_social_dot_events_user_id"):
        op.drop_index("ix_social_dot_events_user_id", table_name="social_dot_events")
    op.drop_table("social_dot_events")
