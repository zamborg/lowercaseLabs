from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    display_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(32), default="dev")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AuditMessage(Base):
    __tablename__ = "audit_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    category: Mapped[str] = mapped_column(String(48), index=True)
    message: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

