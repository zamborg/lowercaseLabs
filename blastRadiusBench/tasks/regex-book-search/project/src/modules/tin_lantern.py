"""Exact-match search logic hidden behind another odd file name."""

from __future__ import annotations

import re
from re import Pattern


def compile_word_pattern(query: str) -> Pattern[str] | None:
    """Compile a case-insensitive whole-word regex for the query."""
    cleaned = query.strip()
    if not cleaned:
        return None
    escaped = re.escape(cleaned)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def find_matching_line_numbers(lines: list[str], pattern: Pattern[str]) -> list[int]:
    """Return 1-based line numbers that match the given regex pattern."""
    return [
        index
        for index, line in enumerate(lines, start=1)
        if pattern.search(line) is not None
    ]

