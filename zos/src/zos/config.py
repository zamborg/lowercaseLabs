"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _default_migrations_dir() -> Path:
    # Works from editable source tree (zos/src/zos/config.py -> zos/migrations)
    # and in container (/app/migrations).
    candidates = [
        Path(__file__).resolve().parents[2] / "migrations",
        Path.cwd() / "migrations",
        Path("/app/migrations"),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    zos_base_url: str = "http://localhost:8080"
    zos_secret_key: str = ""
    zos_owner_password_hash: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    gcal_backend: str = ""
    gcal_mcp_url: str = ""
    migrations_dir: Path = field(default_factory=_default_migrations_dir)

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> Settings:
        if load_env_file:
            load_dotenv(override=False)
        database_url = os.environ.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        base_url = os.environ.get("ZOS_BASE_URL", "http://localhost:8080").strip().rstrip("/")
        mig_override = os.environ.get("ZOS_MIGRATIONS_DIR", "").strip()
        return cls(
            database_url=database_url,
            zos_base_url=base_url,
            zos_secret_key=os.environ.get("ZOS_SECRET_KEY", "").strip(),
            zos_owner_password_hash=os.environ.get("ZOS_OWNER_PASSWORD_HASH", "").strip(),
            google_client_id=os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
            google_client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", "").strip(),
            gcal_backend=os.environ.get("ZOS_GCAL_BACKEND", "").strip().lower(),
            gcal_mcp_url=os.environ.get("ZOS_GCAL_MCP_URL", "").strip(),
            migrations_dir=Path(mig_override) if mig_override else _default_migrations_dir(),
        )
