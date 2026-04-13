"""Experimental code that is unrelated to the exact-match task."""

from __future__ import annotations


def rank_by_overlap(query: str, candidates: list[str]) -> list[str]:
    """Return candidates ordered by naive character overlap."""
    query_set = set(query.lower())
    return sorted(
        candidates,
        key=lambda candidate: sum(ch in query_set for ch in candidate.lower()),
        reverse=True,
    )

