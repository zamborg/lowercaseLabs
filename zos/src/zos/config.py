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


def _default_catalog_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "catalog" / "connectors",
        Path.cwd() / "catalog" / "connectors",
        Path("/app/catalog/connectors"),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _default_modules_catalog_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "catalog" / "modules",
        Path.cwd() / "catalog" / "modules",
        Path("/app/catalog/modules"),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _parse_host_list(raw: str) -> tuple[str, ...]:
    return tuple(h.strip().lower().rstrip(".") for h in raw.split(",") if h.strip())


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_positive_float(raw: str, default: float) -> float:
    val = raw.strip()
    if not val:
        return default
    parsed = float(val)
    if parsed <= 0:
        raise ValueError(f"Expected positive number, got {val!r}")
    return parsed


def _parse_positive_int(raw: str, default: int) -> int:
    val = raw.strip()
    if not val:
        return default
    parsed = int(val)
    if parsed <= 0:
        raise ValueError(f"Expected positive integer, got {val!r}")
    return parsed


def _parse_non_negative_float(raw: str, default: float) -> float:
    val = raw.strip()
    if not val:
        return default
    parsed = float(val)
    if parsed < 0:
        raise ValueError(f"Expected non-negative number, got {val!r}")
    return parsed


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
    allowed_redirect_hosts: tuple[str, ...] = ()
    dcr_max_per_hour: int = 10
    max_oauth_clients: int = 50
    approval_wait_seconds: float = 45.0
    approval_ttl_seconds: int = 600
    policy_engines: tuple[str, ...] = ()
    policy_engine_timeout_seconds: float = 10.0
    module_host: str = ""
    docker_socket: str = "/var/run/docker.sock"
    module_memory_bytes: int = 268435456
    module_network_internal: str = "zos_modules"
    module_network_egress: str = "zos_modules_egress"
    module_container_prefix: str = "zos-mod"
    instructions_file: str = ""
    migrations_dir: Path = field(default_factory=_default_migrations_dir)
    catalog_dir: Path = field(default_factory=_default_catalog_dir)
    modules_catalog_dir: Path = field(default_factory=_default_modules_catalog_dir)

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> Settings:
        if load_env_file:
            load_dotenv(override=False)
        database_url = os.environ.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        base_url = os.environ.get("ZOS_BASE_URL", "http://localhost:8080").strip().rstrip("/")
        mig_override = os.environ.get("ZOS_MIGRATIONS_DIR", "").strip()
        cat_override = os.environ.get("ZOS_CATALOG_DIR", "").strip()
        mod_cat_override = os.environ.get("ZOS_MODULES_CATALOG_DIR", "").strip()
        mod_host = os.environ.get("ZOS_MODULE_HOST", "").strip().lower()
        if mod_host and mod_host not in ("docker", "manual"):
            raise ValueError(f"ZOS_MODULE_HOST must be 'docker' or 'manual', got {mod_host!r}")
        return cls(
            database_url=database_url,
            zos_base_url=base_url,
            zos_secret_key=os.environ.get("ZOS_SECRET_KEY", "").strip(),
            zos_owner_password_hash=os.environ.get("ZOS_OWNER_PASSWORD_HASH", "").strip(),
            google_client_id=os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
            google_client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", "").strip(),
            gcal_backend=os.environ.get("ZOS_GCAL_BACKEND", "").strip().lower(),
            gcal_mcp_url=os.environ.get("ZOS_GCAL_MCP_URL", "").strip(),
            allowed_redirect_hosts=_parse_host_list(
                os.environ.get("ZOS_ALLOWED_REDIRECT_HOSTS", "")
            ),
            dcr_max_per_hour=_parse_positive_int(os.environ.get("ZOS_DCR_MAX_PER_HOUR", ""), 10),
            max_oauth_clients=_parse_positive_int(os.environ.get("ZOS_MAX_OAUTH_CLIENTS", ""), 50),
            approval_wait_seconds=_parse_non_negative_float(
                os.environ.get("ZOS_APPROVAL_WAIT_S", ""), 45.0
            ),
            approval_ttl_seconds=_parse_positive_int(
                os.environ.get("ZOS_APPROVAL_TTL_S", ""), 600
            ),
            policy_engines=_parse_csv_list(os.environ.get("ZOS_POLICY_ENGINES", "")),
            policy_engine_timeout_seconds=_parse_positive_float(
                os.environ.get("ZOS_POLICY_ENGINE_TIMEOUT_S", ""), 10.0
            ),
            module_host=mod_host,
            docker_socket=os.environ.get("ZOS_DOCKER_SOCKET", "/var/run/docker.sock").strip()
            or "/var/run/docker.sock",
            module_memory_bytes=_parse_positive_int(
                os.environ.get("ZOS_MODULE_MEMORY_BYTES", ""), 268435456
            ),
            module_network_internal=(
                os.environ.get("ZOS_MODULE_NETWORK_INTERNAL", "zos_modules").strip()
                or "zos_modules"
            ),
            module_network_egress=(
                os.environ.get("ZOS_MODULE_NETWORK_EGRESS", "zos_modules_egress").strip()
                or "zos_modules_egress"
            ),
            module_container_prefix=(
                os.environ.get("ZOS_MODULE_CONTAINER_PREFIX", "zos-mod").strip()
                or "zos-mod"
            ),
            instructions_file=os.environ.get("ZOS_INSTRUCTIONS_FILE", "").strip(),
            migrations_dir=Path(mig_override) if mig_override else _default_migrations_dir(),
            catalog_dir=Path(cat_override) if cat_override else _default_catalog_dir(),
            modules_catalog_dir=(
                Path(mod_cat_override) if mod_cat_override else _default_modules_catalog_dir()
            ),
        )
