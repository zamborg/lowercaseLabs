"""Add reversible account decommission table.

Revision ID: 0002_account_decommissions
Revises: 0001_initial
Create Date: 2026-02-18 00:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002_account_decommissions"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(index_name == index["name"] for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    if not _table_exists("account_decommissions"):
        op.create_table(
            "account_decommissions",
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("reason", sa.String(length=255), nullable=True),
            sa.Column("decommissioned_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("user_id"),
        )

    if not _index_exists("account_decommissions", "ix_account_decommissions_decommissioned_at"):
        op.create_index(
            "ix_account_decommissions_decommissioned_at",
            "account_decommissions",
            ["decommissioned_at"],
            unique=False,
        )


def downgrade() -> None:
    if _table_exists("account_decommissions"):
        if _index_exists("account_decommissions", "ix_account_decommissions_decommissioned_at"):
            op.drop_index("ix_account_decommissions_decommissioned_at", table_name="account_decommissions")
        op.drop_table("account_decommissions")

