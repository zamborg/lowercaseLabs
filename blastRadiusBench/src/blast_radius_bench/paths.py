"""Path normalization helpers."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from urllib.parse import urlparse

DEFAULT_REPO_ROOT_ALIASES = ("/workspace", "/app", "/repo")

_LINE_REF_PATTERN = re.compile(r"^(?P<path>.+?):\d+(?::\d+)?$")
_HASH_LINE_PATTERN = re.compile(r"^(?P<path>.+?)#L\d+(?:C\d+)?$")


def normalize_repo_path(
    raw: str | None,
    repo_root_aliases: Sequence[str] = DEFAULT_REPO_ROOT_ALIASES,
    *,
    allow_bare: bool = False,
) -> str | None:
    """Return a repo-relative POSIX path when the input looks like one."""
    if raw is None:
        return None

    value = str(raw).strip().strip("\"'` ,:;()[]{}")
    if not value or "\n" in value:
        return None
    if value.startswith("-") or value.startswith("$"):
        return None

    if value.startswith("file://"):
        parsed = urlparse(value)
        value = parsed.path
    elif "://" in value:
        return None

    match = _LINE_REF_PATTERN.match(value) or _HASH_LINE_PATTERN.match(value)
    if match:
        value = match.group("path")

    value = value.replace("\\", "/")

    if not allow_bare and "/" not in value and not value.startswith("."):
        if "." not in value:
            return None

    for alias in sorted({alias.rstrip("/") for alias in repo_root_aliases}, key=len, reverse=True):
        if not alias:
            continue
        if value == alias:
            return None
        if value.startswith(f"{alias}/"):
            value = value[len(alias) :]
            break

    if value.startswith("~/"):
        return None

    normalized = posixpath.normpath(value)
    normalized = normalized.lstrip("/")

    if normalized in {"", "."}:
        return None
    if normalized == ".." or normalized.startswith("../"):
        return None

    return normalized


def normalize_spec_path(raw: str) -> str:
    """Normalize a task-spec path and reject invalid values."""
    normalized = normalize_repo_path(raw, (), allow_bare=True)
    if normalized is None:
        raise ValueError(f"Invalid repo-relative path: {raw!r}")
    return normalized

