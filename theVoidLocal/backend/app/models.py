from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, now_utc


class EntryStatus(str, Enum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class RevealMode(str, Enum):
    ANONYMOUS = "anonymous"
    REVEALED_TO_FRIENDS = "revealed_to_friends"
    REVEALED_TO_SPECIFIC = "revealed_to_specific"


class JobType(str, Enum):
    TRANSCRIPTION = "transcription"
    INSIGHTS = "insights"
    SOCIAL_AGGREGATION = "social_aggregation"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    apple_sub: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    anonymous_handle: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    daily_checkin_time_local: Mapped[str] = mapped_column(String(8), default="20:00")
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    notification_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    entries: Mapped[list[Entry]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AccountDecommission(Base):
    __tablename__ = "account_decommissions"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decommissioned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, index=True)


class Entry(Base):
    __tablename__ = "entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    local_date: Mapped[date] = mapped_column(Date, index=True)
    audio_object_key: Mapped[str] = mapped_column(String(512), unique=True)
    duration_seconds: Mapped[int] = mapped_column(Integer)
    status: Mapped[EntryStatus] = mapped_column(SAEnum(EntryStatus), default=EntryStatus.UPLOADING, index=True)
    trace_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    user: Mapped[User] = relationship(back_populates="entries")
    transcript: Mapped[Transcript | None] = relationship(back_populates="entry", uselist=False, cascade="all, delete-orphan")
    insight: Mapped[Insight | None] = relationship(back_populates="entry", uselist=False, cascade="all, delete-orphan")


class Transcript(Base):
    __tablename__ = "transcripts"

    entry_id: Mapped[str] = mapped_column(String(36), ForeignKey("entries.id", ondelete="CASCADE"), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    provider_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    entry: Mapped[Entry] = relationship(back_populates="transcript")


class Insight(Base):
    __tablename__ = "insights"

    entry_id: Mapped[str] = mapped_column(String(36), ForeignKey("entries.id", ondelete="CASCADE"), primary_key=True)
    mood_score: Mapped[float] = mapped_column(Float)
    mood_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text)
    themes: Mapped[list[str]] = mapped_column(JSON, default=list)
    signals: Mapped[dict] = mapped_column(JSON, default=dict)
    safety_flags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    entry: Mapped[Entry] = relationship(back_populates="insight")


class FriendEdge(Base):
    __tablename__ = "friend_edges"
    __table_args__ = (UniqueConstraint("user_id", "friend_user_id", name="uq_friend_edge"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    friend_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class SocialPresence(Base):
    __tablename__ = "social_presence"
    __table_args__ = (UniqueConstraint("user_id", "local_date", name="uq_social_presence_user_day"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    local_date: Mapped[date] = mapped_column(Date, index=True)
    dot_color: Mapped[str] = mapped_column(String(32))
    dot_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    reveal_mode: Mapped[RevealMode] = mapped_column(SAEnum(RevealMode), default=RevealMode.ANONYMOUS, index=True)
    reveal_friend_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    display_name_override: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)


class InviteToken(Base):
    __tablename__ = "invite_tokens"

    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    inviter_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    max_uses: Mapped[int] = mapped_column(Integer, default=25)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[JobType] = mapped_column(SAEnum(JobType), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), default=JobStatus.PENDING, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    trace_id: Mapped[str] = mapped_column(String(36), index=True)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class FeedbackReport(Base):
    __tablename__ = "feedback_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, index=True)
