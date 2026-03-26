from __future__ import annotations

import pytest

from regex import search_book


def test_search_book_finds_exact_match_in_moby_dick() -> None:
    assert search_book("moby_dick", "Ishmael") == [1]


def test_search_book_is_case_insensitive_and_normalizes_book_name() -> None:
    assert search_book("sherlock holmes", "THE") == [1, 3]


def test_search_book_requires_whole_word_matches() -> None:
    assert search_book("pride_and_prejudice", "acknowledge") == []
    assert search_book("pride_and_prejudice", "truth") == [1]


def test_search_book_returns_empty_list_for_blank_query() -> None:
    assert search_book("moby_dick", "   ") == []


def test_search_book_raises_for_missing_book() -> None:
    with pytest.raises(FileNotFoundError):
        search_book("little_women", "march")
