from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args: dict[str, object] = {}
if settings.resolved_database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(settings.resolved_database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_storage_directories() -> None:
    settings.absolute_storage_root.mkdir(parents=True, exist_ok=True)
    (settings.absolute_storage_root / "audio").mkdir(parents=True, exist_ok=True)
