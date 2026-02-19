"""Add dot_tags payload for social presence.

Revision ID: 0003_social_presence_dot_tags
Revises: 0002_account_decommissions
Create Date: 2026-02-19 00:10:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_social_presence_dot_tags"
down_revision = "0002_account_decommissions"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    if not _table_exists("social_presence"):
        return

    if _column_exists("social_presence", "dot_tags"):
        return

    bind = op.get_bind()
    server_default = sa.text("'[]'::json") if bind.dialect.name == "postgresql" else sa.text("'[]'")

    op.add_column(
        "social_presence",
        sa.Column("dot_tags", sa.JSON(), nullable=False, server_default=server_default),
    )
    op.alter_column("social_presence", "dot_tags", server_default=None)


def downgrade() -> None:
    if not _table_exists("social_presence"):
        return
    if _column_exists("social_presence", "dot_tags"):
        op.drop_column("social_presence", "dot_tags")
