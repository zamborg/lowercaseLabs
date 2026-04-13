"""A distractor helper that the task does not need."""

from __future__ import annotations


def reverse_index(values: list[str]) -> dict[str, int]:
    return {value[::-1]: index for index, value in enumerate(values)}

