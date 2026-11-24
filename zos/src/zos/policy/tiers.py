"""Tool tier resolution: longest literal match wins, unknown -> external."""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Literal

Tier = Literal["read", "write", "external"]

_WILDCARD_CHARS = frozenset("*?[]")


def pattern_specificity(pattern: str) -> tuple[int, int, int]:
    """Rank matching patterns so exact & longest-literal matches always win."""
    is_exact = 1 if not any(ch in _WILDCARD_CHARS for ch in pattern) else 0
    literal_len = sum(1 for ch in pattern if ch not in _WILDCARD_CHARS)
    return (literal_len, is_exact, len(pattern))


def match_best_pattern[T](tool_name: str, rules: Mapping[str, T]) -> T | None:
    """Return the value for the highest-specificity pattern matching `tool_name`, or None."""
    best_key: tuple[int, int, int] | None = None
    best_val: T | None = None
    for pattern, value in rules.items():
        if fnmatchcase(tool_name, pattern):
            spec = pattern_specificity(pattern)
            if best_key is None or spec > best_key:
                best_key = spec
                best_val = value
    return best_val


def resolve_tier(tool_name: str, tiers: Mapping[str, Tier]) -> Tier:
    """Resolve a tool's global tier from `tool_tiers` rows; unknown -> 'external'."""
    matched = match_best_pattern(tool_name, tiers)
    return matched if matched is not None else "external"
