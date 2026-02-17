import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    encryption_key: str
    default_signature: str
    poll_interval: float


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/agentic_mail",
        ),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        encryption_key=os.environ.get("ENCRYPTION_KEY", "dev-insecure-key"),
        default_signature=os.environ.get("DEFAULT_SIGNATURE", "Best"),
        poll_interval=float(os.environ.get("AGENT_POLL_INTERVAL", "2")),
    )
