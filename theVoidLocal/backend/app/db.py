from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _normalize_database_url(url: str) -> str:
    # Fly's Postgres attach injects "postgresql://", but this codebase uses psycopg v3.
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def _connect_args_for(url: str) -> dict[str, bool]:
    return {"check_same_thread": False} if url.startswith("sqlite") else {}


database_url = _normalize_database_url(settings.database_url)
engine = create_engine(database_url, future=True, connect_args=_connect_args_for(database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

_admin_database_url_raw = (settings.admin_database_url or "").strip()
admin_database_url = _normalize_database_url(_admin_database_url_raw) if _admin_database_url_raw else database_url
admin_engine = create_engine(
    admin_database_url,
    future=True,
    connect_args=_connect_args_for(admin_database_url),
)
AdminSessionLocal = sessionmaker(bind=admin_engine, autoflush=False, autocommit=False, future=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime for safe comparisons."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_admin_db() -> Generator[Session, None, None]:
    db = AdminSessionLocal()
    try:
        yield db
    finally:
        db.close()
