#!/bin/bash

set -euo pipefail

cat > /app/src/regex.py <<'EOF'
"""Small exact-word search entrypoint for the books corpus."""

from __future__ import annotations

from modules.fold_map import read_book_lines, resolve_book_path
from modules.tin_lantern import compile_word_pattern, find_matching_line_numbers


def search_book(book_name: str, query: str) -> list[int]:
    """Return 1-based line numbers whose text contains the query as a whole word."""
    pattern = compile_word_pattern(query)
    if pattern is None:
        return []

    book_path = resolve_book_path(book_name)
    lines = read_book_lines(book_path)
    return find_matching_line_numbers(lines, pattern)
EOF
