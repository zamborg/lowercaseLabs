"""Tool naming validation and deterministic <=48-char namespacing."""

from __future__ import annotations

import hashlib
import re

MAX_TOOL_NAME_LENGTH = 48
RESERVED_PREFIXES = frozenset({"zos"})
_PREFIX_RE = re.compile(r"^[a-zA-Z0-9-]+$")
_TOOL_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


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
