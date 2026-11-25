"""Schema-validated connector catalog loader (`catalog/connectors/*.toml`).

Each catalog entry is the `runtime.kind = "remote"` case of the future V2 module manifest.
Malformed entries are logged and skipped so a bad manifest never breaks startup.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from zos.policy.tiers import Tier
from zos.providers.upstream_manager import validate_upstream_url
from zos.router.naming import BUILTIN_PROVIDER_SLUGS, validate_upstream_slug

logger = logging.getLogger(__name__)


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
