from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, model_validator

from sovereign_store.models.app import NetworkingMode


class AppCreate(BaseModel):
    slug: str = Field(min_length=3, max_length=64, pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=2, max_length=100)
    docker_image: str = Field(min_length=3, max_length=255)
    fly_app_name: str = Field(min_length=3, max_length=64)
    networking_mode: NetworkingMode
    internal_port: int = Field(default=8080, ge=1, le=65535)
    requires_volume: bool = False


class AppRead(BaseModel):
    id: UUID
    slug: str
    name: str
    docker_image: str
    fly_app_name: str
    networking_mode: NetworkingMode
    internal_port: int
    requires_volume: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AppChartParseRequest(BaseModel):
    repo_path: str | None = None
    repo_url: HttpUrl | None = None
    ref: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "AppChartParseRequest":
        if bool(self.repo_path) == bool(self.repo_url):
            raise ValueError("Provide exactly one of repo_path or repo_url")
        return self


class AppChartParseResponse(BaseModel):
    parse_mode: str
    repository: str
    dockerfile_found: bool
    fly_toml_found: bool
    fly_app_name: str | None
    networking_mode: NetworkingMode | None
    internal_port: int | None
    requires_volume: bool
    warnings: list[str] = Field(default_factory=list)


class AppCreateFromChartRequest(BaseModel):
    slug: str = Field(min_length=3, max_length=64, pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=2, max_length=100)
    docker_image: str = Field(min_length=3, max_length=255)
    repo_path: str | None = None
    repo_url: HttpUrl | None = None
    ref: str | None = None
    fly_app_name: str | None = Field(default=None, min_length=3, max_length=64)
    networking_mode: NetworkingMode | None = None
    internal_port: int | None = Field(default=None, ge=1, le=65535)
    requires_volume: bool | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "AppCreateFromChartRequest":
        if bool(self.repo_path) == bool(self.repo_url):
            raise ValueError("Provide exactly one of repo_path or repo_url")
        return self


class AppCreateFromChartResponse(BaseModel):
    app: AppRead
    parse: AppChartParseResponse
