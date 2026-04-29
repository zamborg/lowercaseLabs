from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "blackhole"
    environment: str = "development"
    base_dir: Path = Path(__file__).resolve().parents[1]
    database_url: str | None = None
    jwt_secret: str = "dev-secret-change-me-and-make-it-at-least-32-bytes"
    access_token_expiry_hours: int = 24 * 30
    allow_dev_auth: bool = True

    object_storage_root: Path = Path("./storage")
    blackhole_router_model: str = "gpt-4.1-mini"
    blackhole_transcription_model: str = "whisper-1"
    blackhole_auto_route_threshold: float = 0.80
    blackhole_audio_retention_hours: int = 24
    blackhole_search_default_limit: int = 20
    blackhole_search_max_limit: int = 50
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    metrics_enabled: bool = True

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.base_dir / 'storage' / 'blackhole.db').resolve()}"

    @property
    def absolute_storage_root(self) -> Path:
        root = self.object_storage_root
        return root if root.is_absolute() else (self.base_dir / root).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
