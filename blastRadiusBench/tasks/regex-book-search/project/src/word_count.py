"""Unrelated utilities for counting tokens in a book."""

from __future__ import annotations


def count_words(text: str) -> int:
    return len([part for part in text.split() if part])

