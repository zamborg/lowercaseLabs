# ZOS Files Module (`files` v0.2.0)

A shared, POSIX-like filesystem for the owner and all connected agents, backed by Postgres (`mod_files.folders` and `mod_files.files`). One agent creates folders or saves files and another agent — or the owner via `/admin/m/files/` — picks them up.

## Filesystem semantics

- **Real folders:** Folders live in `folders` (root `""` is implicit). Empty folders persist until deleted. Writing, moving, or copying automatically creates missing parent folders (`mkdir -p`). A path cannot be both a file and a folder.
- **Atomic subtree ops:** Moving, copying, or recursively deleting a folder runs in a single Postgres transaction serialized by an advisory lock. Overwriting an existing destination or moving/copying a folder into its own subtree is refused.
- **Quotas & limits:** 10 MiB per file and a configurable total quota (`FILES_QUOTA_BYTES`, default 1 GiB). Both writes and recursive copies count against the quota.

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

1. **Orient quickly with `files_tree`:**
   ```json
   files_tree({"path": "skills", "depth": 3})
   ```
   Returns a compact directory tree (`text` + `entries`, capped so large drives stay concise).
2. **Inspect a folder with `files_list` (or `files_stat`):**
   ```json
   files_list({"path": "skills/my-skill"})
   ```
   Lists direct folders first, then files, with `size`, `sha256` (for files), `updated_at`, and `updated_by`. Compare `sha256` against local files and skip unchanged ones.
3. **Fetch changed or new files with `files_read`:**
   ```json
   files_read({"path": "skills/my-skill/SKILL.md"})
   ```
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
| `files_list(path="", limit=100, cursor?)` | `read` | `ls` of one folder: direct folders first, then files, paginated. |
| `files_tree(path="", depth=3, max_entries=200)` | `read` | Compact depth- and entry-capped tree to orient quickly. |
| `files_stat(path="")` | `read` | Metadata for a file, folder (with child/total counts & size), or root. |
| `files_read(path)` | `read` | Read a file as UTF-8 text or base64 binary, with `sha256`. |
| `files_write(path, content, encoding?, if_sha256?)` | `write` | Create or update a file (`mkdir -p` parents, max 10 MiB, quota-checked). |
| `files_mkdir(path)` | `write` | Create a folder and missing parents (idempotent). |
| `files_move(from, to)` | `write` | Move or rename a file or folder atomically (no overwrite, creates parents). |
| `files_copy(from, to)` | `write` | Copy a file or folder recursively in one transaction (quota-checked). |
| `files_delete(path, recursive=false)` | `write` | Delete a file, empty folder, or (with `recursive=true`) subtree atomically (`destructiveHint`). |

## Owner UI (`/admin/m/files/`)

Mobile-friendly file manager with breadcrumbs (folders listed first), usage vs quota bar, New folder, Upload into the current folder, Rename/move and Copy for files and folders, two-step Delete confirmation showing exact file/subfolder counts for folders, text/markdown preview, and safe `attachment` downloads. Viewing or downloading never mutates `updated_by`.

## Local development

```bash
cd zos/modules/files
uv sync --frozen
uv run ruff check
uv run pytest
```
