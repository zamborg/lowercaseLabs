"""Add feedback reports table.

Revision ID: 0004_feedback_reports
Revises: 0003_social_presence_dot_tags
Create Date: 2026-02-20 00:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0004_feedback_reports"
down_revision = "0003_social_presence_dot_tags"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(index_name == index["name"] for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    if not _table_exists("feedback_reports"):
        op.create_table(
            "feedback_reports",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("kind", sa.String(length=16), nullable=False),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _index_exists("feedback_reports", "ix_feedback_reports_user_id"):
        op.create_index("ix_feedback_reports_user_id", "feedback_reports", ["user_id"], unique=False)
    if not _index_exists("feedback_reports", "ix_feedback_reports_kind"):
        op.create_index("ix_feedback_reports_kind", "feedback_reports", ["kind"], unique=False)
    if not _index_exists("feedback_reports", "ix_feedback_reports_created_at"):
        op.create_index("ix_feedback_reports_created_at", "feedback_reports", ["created_at"], unique=False)


def downgrade() -> None:
    if not _table_exists("feedback_reports"):
        return
    if _index_exists("feedback_reports", "ix_feedback_reports_created_at"):
        op.drop_index("ix_feedback_reports_created_at", table_name="feedback_reports")
    if _index_exists("feedback_reports", "ix_feedback_reports_kind"):
        op.drop_index("ix_feedback_reports_kind", table_name="feedback_reports")
    if _index_exists("feedback_reports", "ix_feedback_reports_user_id"):
        op.drop_index("ix_feedback_reports_user_id", table_name="feedback_reports")
    op.drop_table("feedback_reports")
