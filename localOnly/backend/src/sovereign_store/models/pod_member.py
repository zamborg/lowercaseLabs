from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sovereign_store.core.database import Base


class PodMemberRole(str, enum.Enum):
    OWNER = "owner"
    GUEST = "guest"


class PodMember(Base):
    __tablename__ = "pod_members"

    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pods.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    role: Mapped[PodMemberRole] = mapped_column(
        SAEnum(PodMemberRole, name="pod_member_role"), nullable=False, default=PodMemberRole.GUEST
    )

    pod = relationship("Pod", back_populates="members")
    user = relationship("User", back_populates="memberships")
