import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    dev_token: str
    dev_user_email: str
    encryption_key: str
    default_signature: str
    api_host: str
    api_port: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/agentic_mail",
        ),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        dev_token=os.environ.get("DEV_TOKEN", "dev-token"),
        dev_user_email=os.environ.get("DEV_USER_EMAIL", "dev@local"),
        encryption_key=os.environ.get("ENCRYPTION_KEY", "dev-insecure-key"),
        default_signature=os.environ.get("DEFAULT_SIGNATURE", "Best"),
        api_host=os.environ.get("API_HOST", "0.0.0.0"),
        api_port=int(os.environ.get("API_PORT", "8000")),
    )
