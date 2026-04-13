"""Path and file helpers hidden behind an unhelpful name."""

from __future__ import annotations

from pathlib import Path

BOOKS_DIR = Path(__file__).resolve().parents[2] / "assets" / "books"


def normalize_book_name(book_name: str) -> str:
    """Convert a user-supplied title into the repository filename convention."""
    return book_name.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_book_path(book_name: str) -> Path:
    """Return the on-disk path for the requested book."""
    path = BOOKS_DIR / f"{normalize_book_name(book_name)}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Unknown book: {book_name}")
    return path


def read_book_lines(book_path: Path) -> list[str]:
    """Read the book file and return individual lines."""
    return book_path.read_text(encoding="utf-8").splitlines()

