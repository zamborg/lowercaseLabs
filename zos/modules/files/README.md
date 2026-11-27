# ZOS Files Module (`files`)

A shared, path-based drive for the owner and all connected agents, stored in Postgres (`mod_files.files`). One agent saves a file and another agent — or the owner via `/admin/m/files/` — picks it up.

## Skills sync convention

Store shared agent skills under `skills/<name>/`:

```text
skills/
  <name>/
    SKILL.md            # main skill instructions
    scripts/...         # optional helper scripts
    references/...      # optional reference docs or assets
```

### Pulling skills onto a machine

1. List everything under `skills/`:
   ```json
   files_list({"prefix": "skills/", "recursive": true})
   ```
   Each entry includes `path`, `size`, `sha256`, `updated_at`, and `updated_by`.
2. Compare each entry's `sha256` against your local file hash (skip unchanged files).
3. Fetch changed or new files with `files_read({"path": "skills/<name>/SKILL.md"})`.
   - UTF-8 text files return `{"encoding": "utf-8", "content": "...", "sha256": "..."}`.
   - Binary files return `{"encoding": "base64", "content": "...", "sha256": "..."}`.

### Pushing edits safely (optimistic concurrency)

Pass the `sha256` you last read as `if_sha256` so two machines syncing at the same time never silently overwrite each other:

```json
files_write({
  "path": "skills/my-skill/SKILL.md",
  "content": "# My skill\n...",
  "encoding": "utf-8",
  "if_sha256": "<sha256-from-last-read>"
})
```

If another agent or the owner edited the file first, `files_write` refuses with `CONFLICT` (including the current `sha256`) so you can re-read and merge. Omit `if_sha256` when creating a new file or intentionally overwriting.

## Tools & tiers

| Tool | Tier | Purpose |
|---|---|---|
| `files_list(prefix?, recursive?, limit, cursor)` | `read` | List files (or direct children + folders when `recursive=false`). |
| `files_read(path)` | `read` | Read a file as UTF-8 text or base64 binary, with `sha256`. |
| `files_write(path, content, encoding?, if_sha256?)` | `write` | Create or update a file (max 10 MiB per file, subject to total quota). |
| `files_delete(path)` | `write` | Delete a single file. |
| `files_move(from, to)` | `write` | Move or rename a file (refuses to overwrite an existing destination). |

## Limits & configuration

- **Per-file limit:** 10 MiB (enforced for both MCP `files_write` and UI multipart uploads).
- **Total quota:** `FILES_QUOTA_BYTES` (optional module config, default `1073741824` = 1 GiB).
- **Path rules:** relative, `/`-separated, no leading `/`, no `..`, no empty segments or control characters, max 512 chars. Folders are implicit path prefixes.
- **Owner UI:** `/admin/m/files/` provides a mobile-friendly folder browser with breadcrumbs, usage vs quota bar, multipart upload, text/markdown preview, safe attachment downloads, rename/move, and two-step delete. Browsing or downloading never changes `updated_by`.

## Local development

```bash
cd zos/modules/files
uv sync --frozen
uv run ruff check
uv run pytest
```
