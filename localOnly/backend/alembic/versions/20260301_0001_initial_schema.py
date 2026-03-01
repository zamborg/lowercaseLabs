"""Initial sovereign store schema

Revision ID: 20260301_0001
Revises:
Create Date: 2026-03-01 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260301_0001"
down_revision = None
branch_labels = None
depends_on = None


networking_mode = postgresql.ENUM(
    "proxy", "direct", name="networking_mode", create_type=False
)
pod_status = postgresql.ENUM(
    "provisioning", "active", "suspended", name="pod_status", create_type=False
)
pod_member_role = postgresql.ENUM(
    "owner", "guest", name="pod_member_role", create_type=False
)


def upgrade() -> None:
    networking_mode.create(op.get_bind(), checkfirst=True)
    pod_status.create(op.get_bind(), checkfirst=True)
    pod_member_role.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "apps",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("docker_image", sa.String(length=255), nullable=False),
        sa.Column("fly_app_name", sa.String(length=64), nullable=False),
        sa.Column("networking_mode", networking_mode, nullable=False),
        sa.Column("internal_port", sa.Integer(), nullable=False),
        sa.Column("requires_volume", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_apps_slug", "apps", ["slug"], unique=True)

    op.create_table(
        "pods",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("app_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fly_machine_id", sa.String(length=64), nullable=True),
        sa.Column("internal_ip", sa.String(length=128), nullable=True),
        sa.Column("external_ip", sa.String(length=128), nullable=True),
        sa.Column("direct_url", sa.String(length=512), nullable=True),
        sa.Column("status", pod_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["app_id"], ["apps.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pods_app_id", "pods", ["app_id"], unique=False)
    op.create_index("ix_pods_owner_id", "pods", ["owner_id"], unique=False)
    op.create_index("ix_pods_fly_machine_id", "pods", ["fly_machine_id"], unique=False)

    op.create_table(
        "pod_members",
        sa.Column("pod_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", pod_member_role, nullable=False),
        sa.ForeignKeyConstraint(["pod_id"], ["pods.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("pod_id", "user_id"),
    )
    op.create_index("ix_pod_members_pod_id", "pod_members", ["pod_id"], unique=False)
    op.create_index("ix_pod_members_user_id", "pod_members", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_pod_members_user_id", table_name="pod_members")
    op.drop_index("ix_pod_members_pod_id", table_name="pod_members")
    op.drop_table("pod_members")

    op.drop_index("ix_pods_fly_machine_id", table_name="pods")
    op.drop_index("ix_pods_owner_id", table_name="pods")
    op.drop_index("ix_pods_app_id", table_name="pods")
    op.drop_table("pods")

    op.drop_index("ix_apps_slug", table_name="apps")
    op.drop_table("apps")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    pod_member_role.drop(op.get_bind(), checkfirst=True)
    pod_status.drop(op.get_bind(), checkfirst=True)
    networking_mode.drop(op.get_bind(), checkfirst=True)
