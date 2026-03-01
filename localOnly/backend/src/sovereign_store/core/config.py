from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/sovereign_store"
    )
    redis_url: str | None = None

    jwt_algorithm: str = "RS256"
    jwt_exp_minutes: int = 60
    jwt_private_key: str | None = None
    jwt_public_key: str | None = None
    jwt_secret: str | None = None

    control_plane_url: str = "http://localhost:8000"
    edge_router_url: str = "http://localhost:8001"
    cors_allowed_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:4173,http://127.0.0.1:4173,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8081,http://127.0.0.1:8081"
    )

    fly_api_base_url: str = "https://api.machines.dev"
    fly_api_token: str | None = None
    fly_default_region: str = "sjc"
    fly_org_slug: str | None = None
    developer_repo_root: str | None = None
    repo_clone_root: str = "/tmp/sovereign-chart-clones"
    repo_clone_timeout_seconds: int = 120
    repo_allowed_hosts: str | None = None

    gateway_session_cookie_name: str = "sovereign_gateway_token"
    gateway_session_cookie_domain: str | None = None
    gateway_session_cookie_secure: bool = False
    gateway_session_cookie_samesite: str = "lax"
    gateway_session_cookie_max_age_seconds: int = 3600
    gateway_session_cookie_path: str = "/"

    machine_cpu_kind: str = "shared"
    machine_cpus: int = 1
    machine_memory_mb: int = 256
    machine_volume_size_gb: int = 1

    @property
    def jwt_signing_key(self) -> str:
        if self.jwt_algorithm.startswith("HS"):
            if not self.jwt_secret:
                raise ValueError("JWT_SECRET is required for HMAC algorithms")
            return self.jwt_secret
        if not self.jwt_private_key:
            raise ValueError("JWT_PRIVATE_KEY is required for asymmetric algorithms")
        return self.jwt_private_key.replace("\\n", "\n")

    @property
    def jwt_verification_key(self) -> str:
        if self.jwt_algorithm.startswith("HS"):
            if not self.jwt_secret:
                raise ValueError("JWT_SECRET is required for HMAC algorithms")
            return self.jwt_secret
        if self.jwt_public_key:
            return self.jwt_public_key.replace("\\n", "\n")
        if self.jwt_private_key:
            return self.jwt_private_key.replace("\\n", "\n")
        raise ValueError(
            "JWT_PUBLIC_KEY or JWT_PRIVATE_KEY is required for asymmetric algorithms"
        )

    @staticmethod
    def _parse_csv(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return self._parse_csv(self.cors_allowed_origins)

    @property
    def allowed_repo_hosts(self) -> set[str]:
        return set(self._parse_csv(self.repo_allowed_hosts))


@lru_cache
def get_settings() -> Settings:
    return Settings()
