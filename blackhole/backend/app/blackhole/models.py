from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Enum as SAEnum, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.models import utcnow


class CaptureMode(str, Enum):
    auto = "auto"
    note = "note"
    todo = "todo"
    search = "search"


class CaptureRecordMode(str, Enum):
    note = "note"
    todo = "todo"


class RoutingTarget(str, Enum):
    note = "note"
    todo = "todo"
    search = "search"


class TodoPriority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"


class TodoStatus(str, Enum):
    open = "open"
    completed = "completed"
    canceled = "canceled"


class TranscriptionSessionStatus(str, Enum):
    created = "created"
    uploading = "uploading"
    transcribing = "transcribing"
    completed = "completed"
    failed = "failed"


class BlackholeNote(Base):
    __tablename__ = "blackhole_notes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    body_markdown: Mapped[str] = mapped_column(Text())
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_capture_id: Mapped[str | None] = mapped_column(ForeignKey("blackhole_captures.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)


class BlackholeTodo(Base):
    __tablename__ = "blackhole_todos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    description_markdown: Mapped[str] = mapped_column(Text(), default="")
    priority: Mapped[TodoPriority] = mapped_column(SAEnum(TodoPriority), default=TodoPriority.normal)
    status: Mapped[TodoStatus] = mapped_column(SAEnum(TodoStatus), default=TodoStatus.open, index=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_capture_id: Mapped[str | None] = mapped_column(ForeignKey("blackhole_captures.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BlackholeCapture(Base):
    __tablename__ = "blackhole_captures"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    mode: Mapped[CaptureRecordMode] = mapped_column(SAEnum(CaptureRecordMode))
    audio_object_key: Mapped[str] = mapped_column(String(512))
    transcript_text: Mapped[str] = mapped_column(Text())
    transcript_provider: Mapped[str] = mapped_column(String(64))
    routing_target: Mapped[RoutingTarget] = mapped_column(SAEnum(RoutingTarget))
    routing_confidence: Mapped[float] = mapped_column(Float())
    routing_payload: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_record_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class BlackholeTranscriptionSession(Base):
    __tablename__ = "blackhole_transcription_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    mode: Mapped[CaptureMode] = mapped_column(SAEnum(CaptureMode), default=CaptureMode.auto)
    status: Mapped[TranscriptionSessionStatus] = mapped_column(
        SAEnum(TranscriptionSessionStatus),
        default=TranscriptionSessionStatus.created,
        index=True,
    )
    audio_object_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    final_transcript: Mapped[str | None] = mapped_column(Text(), nullable=True)
    route: Mapped[RoutingTarget | None] = mapped_column(SAEnum(RoutingTarget), nullable=True)
    route_confidence: Mapped[float | None] = mapped_column(Float(), nullable=True)
    route_payload: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    capture_id: Mapped[str | None] = mapped_column(ForeignKey("blackhole_captures.id"), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


Index("ix_blackhole_notes_user_unarchived_updated", BlackholeNote.user_id, BlackholeNote.archived_at, BlackholeNote.updated_at)
Index("ix_blackhole_todos_user_status_updated", BlackholeTodo.user_id, BlackholeTodo.status, BlackholeTodo.updated_at)
Index("ix_blackhole_captures_user_created", BlackholeCapture.user_id, BlackholeCapture.created_at)
Index("ix_blackhole_sessions_user_created", BlackholeTranscriptionSession.user_id, BlackholeTranscriptionSession.created_at)

