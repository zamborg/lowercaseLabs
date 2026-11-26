"""Tool naming validation and deterministic <=48-char namespacing."""

from __future__ import annotations

import hashlib
import re

MAX_TOOL_NAME_LENGTH = 48
RESERVED_PREFIXES = frozenset({"zos"})
BUILTIN_PROVIDER_SLUGS = frozenset({"zos"})
_PREFIX_RE = re.compile(r"^[a-zA-Z0-9-]+$")
_UPSTREAM_SLUG_RE = re.compile(r"^[a-z][a-z0-9]{1,15}$")
_DEVICE_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_TOOL_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_NON_TOOL_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_{2,}")


def validate_device_slug(slug: str) -> str:
    """Validate a device slug (`^[a-z][a-z0-9-]{1,31}$`, no underscores, not `zos`)."""
    cleaned = slug.strip().lower()
    if not _DEVICE_SLUG_RE.fullmatch(cleaned) or cleaned.endswith("-"):
        raise ValueError(
            f"Invalid device slug {slug!r}: must be 2-32 lowercase letters, digits, or hyphens (starting with a letter, no underscores)"
        )
    if cleaned in RESERVED_PREFIXES:
        raise ValueError(f"Slug {cleaned!r} is reserved for router-owned tools")
    return cleaned


def validate_upstream_slug(slug: str) -> str:
    """Validate a user-registered remote MCP server slug (`^[a-z][a-z0-9]{1,15}$`, not `zos`)."""
    cleaned = slug.strip()
    if not _UPSTREAM_SLUG_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid upstream slug {slug!r}: must match ^[a-z][a-z0-9]{{1,15}}$"
        )
    if cleaned in RESERVED_PREFIXES:
        raise ValueError(f"Slug {cleaned!r} is reserved for router-owned tools")
    return cleaned


def sanitize_upstream_tool_name(raw_name: str) -> str:
    """Sanitize an upstream MCP tool name to valid `[a-zA-Z0-9_-]+` characters."""
    cleaned = _NON_TOOL_CHARS_RE.sub("_", raw_name.strip())
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"Upstream tool name {raw_name!r} has no valid characters after sanitization")
    return cleaned


def validate_prefix(prefix: str, *, allow_reserved: bool = False) -> str:
    """Validate provider namespace / tool prefix (no underscores, [a-zA-Z0-9-]+)."""
    if not prefix or not _PREFIX_RE.fullmatch(prefix):
        raise ValueError(
            f"Invalid tool prefix {prefix!r}: must be non-empty [a-zA-Z0-9-]+ with no underscores"
        )
    if not allow_reserved and prefix in RESERVED_PREFIXES:
        raise ValueError(f"Prefix {prefix!r} is reserved for router-owned tools")
    return prefix


def validate_local_tool_name(name: str) -> str:
    """Validate unprefixed local tool name ([a-zA-Z0-9_-]+)."""
    if not name or not _TOOL_RE.fullmatch(name):
        raise ValueError(f"Invalid tool name {name!r}: must match [a-zA-Z0-9_-]+")
    return name


def format_tool_name(prefix: str, local_name: str, *, allow_reserved: bool = False) -> str:
    """Compose `<prefix>_<tool>` capped deterministically at 48 chars with 6-char hash on overflow."""
    validate_prefix(prefix, allow_reserved=allow_reserved)
    validate_local_tool_name(local_name)
    full = f"{prefix}_{local_name}"
    if len(full) <= MAX_TOOL_NAME_LENGTH:
        return full
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:6]
    # 41 + 1 ('_') + 6 (hex) = 48
    return f"{full[: MAX_TOOL_NAME_LENGTH - 7]}_{digest}"


def split_prefix(exposed_name: str) -> tuple[str, str] | None:
    """Split an exposed tool name into (prefix, remainder) on the first underscore."""
    if "_" not in exposed_name:
        return None
    prefix, rest = exposed_name.split("_", 1)
    if not prefix or not rest:
        return None
    return prefix, rest
