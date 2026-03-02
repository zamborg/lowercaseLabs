from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "theVoid-api"
    environment: str = "development"
    database_url: str = "sqlite:///./backend/thevoid.db"
    admin_database_url: str | None = None

    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = 60 * 24 * 30

    upload_token_ttl_seconds: int = 60 * 15
    invite_base_url: str = "thevoid://invite"

    object_storage_root: str = "./backend/storage"
    model_assets_root: str = "/data/model_assets"
    worker_poll_interval_seconds: float = 2.0
    inline_worker_enabled: bool = False

    admin_enabled: bool = True
    admin_username: str = "admin"
    admin_password: str = "admin"

    openai_api_key: str | None = None
    openai_transcription_model: str = "gpt-4o-mini-transcribe"
    openai_transcription_language: str | None = None
    openai_insights_model: str = "gpt-4.1-mini"
    openai_request_timeout_seconds: float = 90.0

    apple_allowed_audiences: str = ""
    apple_issuer: str = "https://appleid.apple.com"
    apple_jwks_url: str = "https://appleid.apple.com/auth/keys"
    apple_jwks_cache_ttl_seconds: int = 60 * 60
    apple_jwks_timeout_seconds: float = 5.0
    allow_dev_identity_tokens: bool = True

    auto_create_schema: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.is_production and self.is_sqlite:
            raise ValueError(
                "DATABASE_URL must point to Postgres in production; sqlite is not supported for Fly deploys."
            )
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"production", "prod"}

    @property
    def apple_audiences(self) -> tuple[str, ...]:
        return tuple(
            value.strip()
            for value in self.apple_allowed_audiences.split(",")
            if value.strip()
        )

    @property
    def dev_identity_fallback_enabled(self) -> bool:
        if self.is_production:
            return False
        return self.allow_dev_identity_tokens


settings = Settings()
