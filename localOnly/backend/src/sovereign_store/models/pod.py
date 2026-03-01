from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sovereign_store.core.database import Base
from sovereign_store.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class PodStatus(str, enum.Enum):
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class Pod(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "pods"

    app_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("apps.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    fly_machine_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    internal_ip: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_ip: Mapped[str | None] = mapped_column(String(128), nullable=True)
    direct_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[PodStatus] = mapped_column(
        SAEnum(PodStatus, name="pod_status"), nullable=False, default=PodStatus.PROVISIONING
    )

    app = relationship("App", back_populates="pods")
    owner = relationship("User", back_populates="owned_pods")
    members = relationship("PodMember", back_populates="pod", cascade="all,delete-orphan")
