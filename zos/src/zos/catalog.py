"""Schema-validated connector catalog loader (`catalog/connectors/*.toml`).

Each catalog entry is the `runtime.kind = "remote"` case of the future V2 module manifest.
Malformed entries are logged and skipped so a bad manifest never breaks startup.
"""

from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from zos.policy.tiers import Tier
from zos.providers.upstream_manager import validate_upstream_url
from zos.router.naming import BUILTIN_PROVIDER_SLUGS, validate_upstream_slug

logger = logging.getLogger(__name__)

_DIGEST_IMAGE_RE = re.compile(
    r"^(?:[a-zA-Z0-9][a-zA-Z0-9._-]*(?::[0-9]{1,5})?/)?"
    r"[a-zA-Z0-9][a-zA-Z0-9._/-]*(?::[a-zA-Z0-9._-]+)?@sha256:[0-9a-f]{64}$"
)
_CONFIG_FIELD_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class ConnectorRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["remote"] = "remote"
    url: str

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return validate_upstream_url(value)


class ConnectorAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "bearer", "oauth"]
    token_help: str = ""
    requires_client_registration: bool = False
    client_help: str = ""


class ConnectorTiersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: Tier = "external"
    patterns: dict[str, Tier] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_tier_table(cls, data: Any) -> Any:
        """Allow `[tiers]` to declare `default = "..."` alongside flat `"pattern" = "read"` keys."""
        if not isinstance(data, dict):
            return data
        if "patterns" in data and isinstance(data["patterns"], dict):
            return data
        default_tier = data.get("default", "external")
        patterns = {k: v for k, v in data.items() if k != "default"}
        return {"default": default_tier, "patterns": patterns}


class ConnectorManifest(BaseModel):
    """V1 remote connector manifest (`manifest_version = 1`, `runtime.kind = "remote"`)."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[1]
    slug: str
    name: str
    description: str
    docs_url: str
    runtime: ConnectorRuntimeConfig
    auth: ConnectorAuthConfig
    tiers: ConnectorTiersConfig = Field(default_factory=ConnectorTiersConfig)

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, value: str) -> str:
        cleaned = validate_upstream_slug(value)
        if cleaned in BUILTIN_PROVIDER_SLUGS:
            raise ValueError(f"Catalog slug '{cleaned}' collides with built-in provider")
        return cleaned

    @field_validator("name", "description")
    @classmethod
    def _check_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must be non-empty")
        return cleaned

    @field_validator("docs_url")
    @classmethod
    def _check_docs_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not (cleaned.startswith("https://") or cleaned.startswith("http://")):
            raise ValueError("docs_url must start with https:// or http://")
        return cleaned


def default_catalog_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "catalog" / "connectors",
        Path.cwd() / "catalog" / "connectors",
        Path("/app/catalog/connectors"),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def load_manifest_file(path: Path) -> ConnectorManifest:
    """Parse and validate a single TOML connector manifest file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    manifest = ConnectorManifest.model_validate(raw)
    if manifest.slug != path.stem:
        raise ValueError(
            f"Manifest slug '{manifest.slug}' does not match filename '{path.name}'"
        )
    return manifest


def load_catalog(catalog_dir: Path | None = None) -> list[ConnectorManifest]:
    """Load all valid connector manifests from `catalog_dir`, skipping malformed files with warnings."""
    directory = (catalog_dir or default_catalog_dir()).expanduser()
    if not directory.is_dir():
        return []

    manifests: list[ConnectorManifest] = []
    seen_slugs: set[str] = set()
    for path in sorted(directory.glob("*.toml")):
        try:
            manifest = load_manifest_file(path)
        except Exception as exc:
            logger.warning("Skipping malformed connector manifest %s: %s", path.name, exc)
            continue
        if manifest.slug in seen_slugs:
            logger.warning("Skipping duplicate connector slug '%s' in %s", manifest.slug, path.name)
            continue
        seen_slugs.add(manifest.slug)
        manifests.append(manifest)
    return manifests


def get_catalog_entry(slug: str, catalog_dir: Path | None = None) -> ConnectorManifest | None:
    clean = slug.strip().lower()
    for entry in load_catalog(catalog_dir):
        if entry.slug == clean:
            return entry
    return None


# ── Manifest v2: Container Modules (`catalog/modules/*.toml`) ───────────────


class ModuleRuntimeConfig(BaseModel):
    """Runtime specification for a v2 container module."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["container"] = "container"
    image: str | None = None
    build: str | None = None
    mcp_port: int = Field(ge=1, le=65535)
    mcp_path: str = "/mcp"
    ui_port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("image")
    @classmethod
    def _check_image_digest_pin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not _DIGEST_IMAGE_RE.fullmatch(cleaned):
            raise ValueError(
                f"Module image {value!r} must be digest-pinned as '<repo>@sha256:<64-hex>'"
            )
        return cleaned

    @field_validator("build")
    @classmethod
    def _check_build_context(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        posix = PurePosixPath(cleaned)
        if (
            not cleaned
            or posix.is_absolute()
            or ".." in posix.parts
            or len(posix.parts) < 2
            or posix.parts[0] != "modules"
        ):
            raise ValueError(
                f"Module build path {value!r} must be a relative path under 'modules/<slug>'"
            )
        return str(posix)

    @field_validator("mcp_path")
    @classmethod
    def _check_mcp_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.startswith("/"):
            raise ValueError("mcp_path must start with '/'")
        return cleaned

    @model_validator(mode="after")
    def _check_image_or_build(self) -> ModuleRuntimeConfig:
        if bool(self.image) == bool(self.build):
            raise ValueError("Container runtime must set exactly one of 'image' or 'build'")
        return self


class ModuleStorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    postgres: bool = False


class ModuleNetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    egress: bool = False


class ModuleUIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    path: str = "/"

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        cleaned = value.strip() or "/"
        if not cleaned.startswith("/"):
            raise ValueError("ui.path must start with '/'")
        return cleaned


class ModuleConfigField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    secret: bool = False
    help: str = ""
    required: bool = False
    default: str | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not _CONFIG_FIELD_NAME_RE.fullmatch(cleaned):
            raise ValueError(
                f"Invalid module config field name {value!r}; expected UPPER_SNAKE_CASE"
            )
        return cleaned


class ModuleManifest(BaseModel):
    """V2 container module manifest (`manifest_version = 2`, `runtime.kind = "container"`)."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[2]
    slug: str
    name: str
    version: str
    description: str
    docs_url: str = ""
    runtime: ModuleRuntimeConfig
    storage: ModuleStorageConfig = Field(default_factory=ModuleStorageConfig)
    network: ModuleNetworkConfig = Field(default_factory=ModuleNetworkConfig)
    ui: ModuleUIConfig | None = None
    config: list[ModuleConfigField] = Field(default_factory=list)
    tiers: ConnectorTiersConfig = Field(default_factory=ConnectorTiersConfig)

    @field_validator("slug")
    @classmethod
    def _check_slug(cls, value: str) -> str:
        cleaned = validate_upstream_slug(value)
        if cleaned in BUILTIN_PROVIDER_SLUGS:
            raise ValueError(f"Module slug '{cleaned}' collides with built-in provider")
        return cleaned

    @field_validator("name", "version", "description")
    @classmethod
    def _check_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must be non-empty")
        return cleaned

    @field_validator("docs_url")
    @classmethod
    def _check_docs_url(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned and not (cleaned.startswith("https://") or cleaned.startswith("http://")):
            raise ValueError("docs_url must start with https:// or http://")
        return cleaned

    @model_validator(mode="after")
    def _check_unique_config_fields(self) -> ModuleManifest:
        seen: set[str] = set()
        for f in self.config:
            if f.name in seen:
                raise ValueError(f"Duplicate config field '{f.name}'")
            seen.add(f.name)
        return self


def default_modules_catalog_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "catalog" / "modules",
        Path.cwd() / "catalog" / "modules",
        Path("/app/catalog/modules"),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def load_module_manifest_file(path: Path) -> ModuleManifest:
    """Parse and validate a single v2 TOML module manifest file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    manifest = ModuleManifest.model_validate(raw)
    if manifest.slug != path.stem:
        raise ValueError(
            f"Module manifest slug '{manifest.slug}' does not match filename '{path.name}'"
        )
    return manifest


def load_module_catalog(modules_dir: Path | None = None) -> list[ModuleManifest]:
    """Load all valid v2 module manifests from `modules_dir`, skipping malformed files."""
    directory = (modules_dir or default_modules_catalog_dir()).expanduser()
    if not directory.is_dir():
        return []

    manifests: list[ModuleManifest] = []
    seen_slugs: set[str] = set()
    for path in sorted(directory.glob("*.toml")):
        try:
            manifest = load_module_manifest_file(path)
        except Exception as exc:
            logger.warning("Skipping malformed module manifest %s: %s", path.name, exc)
            continue
        if manifest.slug in seen_slugs:
            logger.warning("Skipping duplicate module slug '%s' in %s", manifest.slug, path.name)
            continue
        seen_slugs.add(manifest.slug)
        manifests.append(manifest)
    return manifests


def get_module_catalog_entry(slug: str, modules_dir: Path | None = None) -> ModuleManifest | None:
    clean = slug.strip().lower()
    for entry in load_module_catalog(modules_dir):
        if entry.slug == clean:
            return entry
    return None
