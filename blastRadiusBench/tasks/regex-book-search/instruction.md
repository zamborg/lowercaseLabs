Implement `search_book` in `src/regex.py`.

Requirements:

- `search_book(book_name: str, query: str) -> list[int]`
- Search `assets/books/<book_name>.txt` for case-insensitive exact word matches of `query`.
- Return sorted 1-based line numbers for matching lines.
- If `query` is blank or whitespace-only, return an empty list.
- Normalize book names so inputs like `"moby dick"` and `"moby_dick"` both resolve to `assets/books/moby_dick.txt`.
- Raise `FileNotFoundError` if the requested book does not exist.

Constraints:

- Edit `src/regex.py`.
- Do not change the tests.
- Keep the public function signature exactly as written.

Notes:

- The repository may already contain code you can reuse.
- The working directory is the repository root at `/app`.
