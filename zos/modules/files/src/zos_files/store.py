"""Postgres storage layer for the ZOS Files module (uses only its own schema).

Implements POSIX-like filesystem semantics on top of Postgres:
- Real `folders` rows (root `""` is implicit; empty folders persist until deleted).
- `files` rows store `bytea` content with sha256, size, content_type, updated_by.
- Writing/moving/copying creates missing parent folders (`mkdir -p`).
- Folder move, copy, and recursive delete execute atomically in a single transaction.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB per file
DEFAULT_QUOTA_BYTES = 1024 * 1024 * 1024  # 1 GiB total
MAX_PATH_CHARS = 512
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 500
DEFAULT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 10
DEFAULT_TREE_MAX_ENTRIES = 200
MAX_TREE_MAX_ENTRIES = 1000

# Serializes all writers so quota, collision, and `if_sha256` checks are race-free.
_WRITE_LOCK_KEY = 0x5A4F5346  # "ZOSF"

MIGRATIONS: list[tuple[str, str]] = [
    (
        "0001_init",
        """
        CREATE TABLE IF NOT EXISTS files (
            id BIGSERIAL PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            content BYTEA NOT NULL,
            size BIGINT NOT NULL,
            sha256 TEXT NOT NULL,
            content_type TEXT NOT NULL,
            updated_by TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS files_path_c_idx ON files (path COLLATE "C");
        """,
    ),
    (
        "0002_folders",
        """
        CREATE TABLE IF NOT EXISTS folders (
            path TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS folders_path_c_idx ON folders (path COLLATE "C");
        INSERT INTO folders (path, created_at, updated_at, updated_by)
        SELECT prefix, MIN(created_at), MAX(updated_at), (ARRAY_AGG(updated_by ORDER BY updated_at DESC))[1]
        FROM (
            SELECT
                array_to_string((string_to_array(f.path, '/'))[1:gs.n], '/') AS prefix,
                f.created_at,
                f.updated_at,
                f.updated_by
            FROM files f
            CROSS JOIN LATERAL generate_series(1, array_length(string_to_array(f.path, '/'), 1) - 1) AS gs(n)
        ) sub
        GROUP BY prefix
        ON CONFLICT (path) DO NOTHING;
        """,
    ),
]


class PathError(ValueError):
    """The path (or prefix) violates the path rules."""


class FileTooLargeError(ValueError):
    """A single file exceeds the per-file size limit."""


class QuotaExceededError(ValueError):
    """The write or copy would push total usage past the configured quota."""


class ConflictError(Exception):
    """Optimistic-concurrency, path-collision, or non-empty folder conflict."""

    def __init__(self, message: str, *, current_sha256: str | None = None) -> None:
        super().__init__(message)
        self.current_sha256 = current_sha256


def normalize_path(raw: str) -> str:
    """Validate and return a canonical non-root path (relative, `/`-separated, no `..`, no trailing `/`)."""
    if not isinstance(raw, str):
        raise PathError("path must be a string")
    if not raw:
        raise PathError("path must be non-empty")
    if len(raw) > MAX_PATH_CHARS:
        raise PathError(f"path must be <= {MAX_PATH_CHARS} characters")
    if raw.startswith("/"):
        raise PathError("path must be relative (no leading '/')")
    if "\\" in raw:
        raise PathError("path must use '/' separators (no backslashes)")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise PathError("path must not contain control characters")
    segments = raw.split("/")
    for seg in segments:
        if seg == "":
            raise PathError("path must not contain empty segments (no '//' or trailing '/')")
        if seg in (".", ".."):
            raise PathError("path must not contain '.' or '..' segments")
    return raw


def normalize_entry_path(raw: str) -> str:
    """Normalize a non-root file or folder path (allows a single optional trailing `/`)."""
    if not isinstance(raw, str):
        raise PathError("path must be a string")
    trimmed = raw[:-1] if len(raw) > 1 and raw.endswith("/") else raw
    return normalize_path(trimmed)


def normalize_dir_path(raw: str | None) -> str:
    """Normalize a directory path to either '' (root) or 'a/b' (no trailing slash)."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise PathError("path must be a string")
    trimmed = raw.strip()
    if trimmed in ("", "/"):
        return ""
    if trimmed.endswith("/"):
        trimmed = trimmed[:-1]
    return normalize_path(trimmed)


def normalize_prefix(raw: str | None) -> str:
    """Normalize a folder prefix to either '' (root) or 'a/b/' (trailing slash)."""
    d = normalize_dir_path(raw)
    return f"{d}/" if d else ""


def _ancestors(clean_path: str) -> list[str]:
    segs = clean_path.split("/")
    return ["/".join(segs[:i]) for i in range(1, len(segs))]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guess_content_type(path: str, data: bytes) -> str:
    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        return guessed
    return "text/plain" if is_text(data) else "application/octet-stream"


def is_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def decode_payload(content: str, encoding: str) -> bytes:
    """Decode a tool/UI payload into bytes, refusing oversize input before decoding."""
    enc = (encoding or "utf-8").strip().lower()
    if enc in ("utf-8", "utf8"):
        if len(content) > MAX_FILE_BYTES:  # chars <= bytes, cheap early reject
            raise FileTooLargeError(f"file exceeds the {MAX_FILE_BYTES}-byte per-file limit")
        return content.encode("utf-8")
    if enc == "base64":
        if len(content) > (MAX_FILE_BYTES * 4) // 3 + 8:
            raise FileTooLargeError(f"file exceeds the {MAX_FILE_BYTES}-byte per-file limit")
        try:
            return base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 content: {exc}") from exc
    raise ValueError("encoding must be 'utf-8' or 'base64'")


def _iso(val: Any) -> str | None:
    if isinstance(val, datetime):
        return val.isoformat()
    return None if val is None else str(val)


def _format_size_short(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"  # pragma: no cover


def _format_file_meta(row: dict[str, Any]) -> dict[str, Any]:
    path = str(row["path"])
    out: dict[str, Any] = {
        "type": "file",
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "size": int(row["size"]),
        "sha256": str(row["sha256"]),
        "content_type": str(row["content_type"]),
        "updated_at": _iso(row["updated_at"]),
        "updated_by": str(row["updated_by"]),
    }
    if "created_at" in row:
        out["created_at"] = _iso(row["created_at"])
    return out


_META_COLS = "path, size, sha256, content_type, created_at, updated_at, updated_by"

FailHook = Callable[[str, AsyncConnection[DictRow]], Awaitable[None]]


def create_files_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


class FilesStore:
    """Async Postgres store for POSIX-like files and real folders."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        *,
        quota_bytes: int = DEFAULT_QUOTA_BYTES,
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self.pool = pool
        self.quota_bytes = quota_bytes
        self.max_file_bytes = max_file_bytes
        # Test-only fault-injection hook for verifying transactional atomicity mid-subtree.
        self._fail_hook: FailHook | None = None

    async def init_schema(self) -> list[str]:
        """Run idempotent, versioned schema migrations inside the module's own schema."""
        applied_now: list[str] = []
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files_schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.commit()
            cur = await conn.execute("SELECT version FROM files_schema_migrations")
            already = {str(r["version"]) for r in await cur.fetchall()}
            for version, sql_body in MIGRATIONS:
                if version in already:
                    continue
                async with conn.transaction():
                    for stmt in sql_body.split(";"):
                        s = stmt.strip()
                        if s:
                            await conn.execute(s)
                    await conn.execute(
                        "INSERT INTO files_schema_migrations (version) VALUES (%s) "
                        "ON CONFLICT DO NOTHING",
                        (version,),
                    )
                applied_now.append(version)
        return applied_now

    # ── reads ────────────────────────────────────────────────────────────────

    async def usage(self) -> dict[str, int]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT
                    (SELECT COALESCE(SUM(size), 0)::bigint FROM files) AS used,
                    (SELECT COUNT(*)::bigint FROM files) AS n_files,
                    (SELECT COUNT(*)::bigint FROM folders) AS n_folders
                """
            )
            row = await cur.fetchone()
        assert row is not None
        return {
            "used_bytes": int(row["used"]),
            "file_count": int(row["n_files"]),
            "folder_count": int(row["n_folders"]),
            "quota_bytes": self.quota_bytes,
            "max_file_bytes": self.max_file_bytes,
        }

    async def _folder_stat(
        self,
        conn: AsyncConnection[DictRow],
        clean: str,
        folder_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pfx = f"{clean}/" if clean else ""
        n = len(pfx)
        cur = await conn.execute(
            """
            SELECT
                (SELECT COUNT(*)::bigint FROM folders
                 WHERE starts_with(path, %(pfx)s) AND position('/' in substr(path, %(n)s + 1)) = 0) AS direct_folders,
                (SELECT COUNT(*)::bigint FROM files
                 WHERE starts_with(path, %(pfx)s) AND position('/' in substr(path, %(n)s + 1)) = 0) AS direct_files,
                (SELECT COUNT(*)::bigint FROM folders WHERE starts_with(path, %(pfx)s)) AS total_folders,
                (SELECT COUNT(*)::bigint FROM files WHERE starts_with(path, %(pfx)s)) AS total_files,
                (SELECT COALESCE(SUM(size), 0)::bigint FROM files WHERE starts_with(path, %(pfx)s)) AS total_size,
                (SELECT MAX(updated_at) FROM files WHERE starts_with(path, %(pfx)s)) AS max_file_updated_at
            """,
            {"pfx": pfx, "n": n},
        )
        agg = await cur.fetchone()
        assert agg is not None
        direct_folders = int(agg["direct_folders"])
        direct_files = int(agg["direct_files"])
        total_folders = int(agg["total_folders"])
        total_files = int(agg["total_files"])
        total_size = int(agg["total_size"])

        updated_at = folder_row["updated_at"] if folder_row else None
        max_file_ts = agg["max_file_updated_at"]
        if max_file_ts is not None and (updated_at is None or max_file_ts > updated_at):
            updated_at = max_file_ts

        return {
            "type": "folder",
            "name": clean.rsplit("/", 1)[-1] if clean else "",
            "path": clean,
            "child_count": direct_folders + direct_files,
            "direct_folder_count": direct_folders,
            "direct_file_count": direct_files,
            "folder_count": total_folders,
            "file_count": total_files,
            "total_size": total_size,
            "size": total_size,
            "created_at": _iso(folder_row["created_at"]) if folder_row else None,
            "updated_at": _iso(updated_at),
            "updated_by": str(folder_row["updated_by"]) if folder_row else None,
        }

    async def stat(self, path: str = "") -> dict[str, Any]:
        """Return metadata for a file, a folder, or the implicit root (`""`)."""
        clean = normalize_dir_path(path)
        async with self.pool.connection() as conn:
            if clean == "":
                return await self._folder_stat(conn, "", None)
            cur = await conn.execute(
                f"SELECT {_META_COLS} FROM files WHERE path = %s",
                (clean,),
            )
            frow = await cur.fetchone()
            if frow is not None:
                return _format_file_meta(frow)
            cur = await conn.execute(
                "SELECT path, created_at, updated_at, updated_by FROM folders WHERE path = %s",
                (clean,),
            )
            drow = await cur.fetchone()
            if drow is not None:
                return await self._folder_stat(conn, clean, drow)
        raise KeyError(f"no such file or folder: {clean}")

    async def read(self, path: str) -> tuple[dict[str, Any], bytes]:
        """Return (metadata, raw bytes). Never mutates anything."""
        clean = normalize_path(path)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_META_COLS}, content FROM files WHERE path = %s",
                (clean,),
            )
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"no such file: {clean}")
        return _format_file_meta(row), bytes(row["content"])

    async def list(
        self,
        *,
        path: str | None = "",
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List direct children of a folder (folders first, then files), keyset-paginated."""
        clean = normalize_dir_path(path)
        lim = int(limit)
        if lim < 1 or lim > MAX_LIST_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}")

        after_grp = -1
        after_name = ""
        if cursor:
            if len(cursor) < 2 or cursor[1] != ":" or cursor[0] not in ("d", "f"):
                raise ValueError("invalid cursor")
            after_grp = 0 if cursor[0] == "d" else 1
            after_name = cursor[2:]

        pfx = f"{clean}/" if clean else ""
        n = len(pfx)

        async with self.pool.connection() as conn:
            if clean != "":
                cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (clean,))
                if await cur.fetchone() is None:
                    cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (clean,))
                    if await cur.fetchone() is not None:
                        raise ValueError(f"'{clean}' is a file, not a folder")
                    raise KeyError(f"no such folder: {clean}")

            cur = await conn.execute(
                """
                WITH direct_folders AS (
                    SELECT
                        0 AS grp,
                        substr(d.path, %(n)s + 1) AS name,
                        d.path AS path,
                        COALESCE(agg.total_size, 0)::bigint AS size,
                        COALESCE(agg.file_count, 0)::bigint AS file_count,
                        COALESCE(subd.folder_count, 0)::bigint AS folder_count,
                        NULL::text AS sha256,
                        NULL::text AS content_type,
                        d.created_at AS created_at,
                        GREATEST(d.updated_at, COALESCE(agg.max_updated_at, d.updated_at)) AS updated_at,
                        d.updated_by AS updated_by
                    FROM folders d
                    LEFT JOIN LATERAL (
                        SELECT
                            COALESCE(SUM(f.size), 0)::bigint AS total_size,
                            COUNT(*)::bigint AS file_count,
                            MAX(f.updated_at) AS max_updated_at
                        FROM files f
                        WHERE starts_with(f.path, d.path || '/')
                    ) agg ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT COUNT(*)::bigint AS folder_count
                        FROM folders s
                        WHERE starts_with(s.path, d.path || '/')
                    ) subd ON TRUE
                    WHERE starts_with(d.path, %(pfx)s)
                      AND position('/' in substr(d.path, %(n)s + 1)) = 0
                ),
                direct_files AS (
                    SELECT
                        1 AS grp,
                        substr(f.path, %(n)s + 1) AS name,
                        f.path AS path,
                        f.size AS size,
                        0::bigint AS file_count,
                        0::bigint AS folder_count,
                        f.sha256 AS sha256,
                        f.content_type AS content_type,
                        f.created_at AS created_at,
                        f.updated_at AS updated_at,
                        f.updated_by AS updated_by
                    FROM files f
                    WHERE starts_with(f.path, %(pfx)s)
                      AND position('/' in substr(f.path, %(n)s + 1)) = 0
                ),
                combined AS (
                    SELECT * FROM direct_folders
                    UNION ALL
                    SELECT * FROM direct_files
                )
                SELECT * FROM combined
                WHERE (grp > %(after_grp)s)
                   OR (grp = %(after_grp)s AND name COLLATE "C" > %(after_name)s)
                ORDER BY grp ASC, name COLLATE "C" ASC
                LIMIT %(lim)s
                """,
                {
                    "pfx": pfx,
                    "n": n,
                    "after_grp": after_grp,
                    "after_name": after_name,
                    "lim": lim + 1,
                },
            )
            rows = await cur.fetchall()

        page = rows[:lim]
        entries: list[dict[str, Any]] = []
        for r in page:
            if int(r["grp"]) == 0:
                entries.append(
                    {
                        "type": "folder",
                        "name": str(r["name"]),
                        "path": str(r["path"]),
                        "size": int(r["size"]),
                        "file_count": int(r["file_count"]),
                        "folder_count": int(r["folder_count"]),
                        "updated_at": _iso(r["updated_at"]),
                        "updated_by": str(r["updated_by"]),
                    }
                )
            else:
                entries.append(_format_file_meta(r))

        next_cursor: str | None = None
        if len(rows) > lim and page:
            last = page[-1]
            prefix_tag = "d" if int(last["grp"]) == 0 else "f"
            next_cursor = f"{prefix_tag}:{last['name']}"

        return {"path": clean, "entries": entries, "next_cursor": next_cursor}

    async def tree(
        self,
        *,
        path: str | None = "",
        depth: int = DEFAULT_TREE_DEPTH,
        max_entries: int = DEFAULT_TREE_MAX_ENTRIES,
    ) -> dict[str, Any]:
        """Return a compact, depth- and entry-capped tree for orienting quickly."""
        clean = normalize_dir_path(path)
        dep = int(depth)
        if dep < 1 or dep > MAX_TREE_DEPTH:
            raise ValueError(f"depth must be between 1 and {MAX_TREE_DEPTH}")
        cap = int(max_entries)
        if cap < 1 or cap > MAX_TREE_MAX_ENTRIES:
            raise ValueError(f"max_entries must be between 1 and {MAX_TREE_MAX_ENTRIES}")

        pfx = f"{clean}/" if clean else ""
        n = len(pfx)

        async with self.pool.connection() as conn:
            if clean != "":
                cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (clean,))
                if await cur.fetchone() is None:
                    cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (clean,))
                    if await cur.fetchone() is not None:
                        raise ValueError(f"'{clean}' is a file, not a folder")
                    raise KeyError(f"no such folder: {clean}")

            # Fetch folders & files within depth limit (bounded well above cap for ordering).
            cur = await conn.execute(
                """
                SELECT path, updated_at, updated_by
                FROM folders
                WHERE starts_with(path, %(pfx)s)
                  AND array_length(string_to_array(substr(path, %(n)s + 1), '/'), 1) <= %(dep)s
                ORDER BY path COLLATE "C"
                LIMIT %(fetch_lim)s
                """,
                {"pfx": pfx, "n": n, "dep": dep, "fetch_lim": cap + 1},
            )
            folder_rows = await cur.fetchall()

            cur = await conn.execute(
                """
                SELECT path, size, updated_at, updated_by
                FROM files
                WHERE starts_with(path, %(pfx)s)
                  AND array_length(string_to_array(substr(path, %(n)s + 1), '/'), 1) <= %(dep)s
                ORDER BY path COLLATE "C"
                LIMIT %(fetch_lim)s
                """,
                {"pfx": pfx, "n": n, "dep": dep, "fetch_lim": cap + 1},
            )
            file_rows = await cur.fetchall()

            # Check if any deeper items exist beyond `depth` so callers know.
            cur = await conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM folders
                    WHERE starts_with(path, %(pfx)s)
                      AND array_length(string_to_array(substr(path, %(n)s + 1), '/'), 1) > %(dep)s
                ) OR EXISTS(
                    SELECT 1 FROM files
                    WHERE starts_with(path, %(pfx)s)
                      AND array_length(string_to_array(substr(path, %(n)s + 1), '/'), 1) > %(dep)s
                ) AS has_deeper
                """,
                {"pfx": pfx, "n": n, "dep": dep},
            )
            deeper_row = await cur.fetchone()
            has_deeper = bool(deeper_row and deeper_row["has_deeper"])

        # Group children by parent path so folders come first at every level.
        child_folders: dict[str, list[dict[str, Any]]] = {}
        child_files: dict[str, list[dict[str, Any]]] = {}
        for r in folder_rows:
            p = str(r["path"])
            parent = p.rsplit("/", 1)[0] if "/" in p else ""
            child_folders.setdefault(parent, []).append(r)
        for r in file_rows:
            p = str(r["path"])
            parent = p.rsplit("/", 1)[0] if "/" in p else ""
            child_files.setdefault(parent, []).append(r)

        for lst in child_folders.values():
            lst.sort(key=lambda r: str(r["path"]))
        for lst in child_files.values():
            lst.sort(key=lambda r: str(r["path"]))

        entries: list[dict[str, Any]] = []
        lines: list[str] = [f"{clean}/" if clean else "/"]
        truncated = len(folder_rows) > cap or len(file_rows) > cap

        def _walk(parent: str, level: int) -> None:
            nonlocal truncated
            for r in child_folders.get(parent, []):
                if len(entries) >= cap:
                    truncated = True
                    return
                p = str(r["path"])
                name = p.rsplit("/", 1)[-1]
                entries.append({"type": "folder", "path": p, "name": name, "depth": level})
                lines.append(f"{'  ' * level}{name}/")
                if level < dep:
                    _walk(p, level + 1)
            for r in child_files.get(parent, []):
                if len(entries) >= cap:
                    truncated = True
                    return
                p = str(r["path"])
                name = p.rsplit("/", 1)[-1]
                size = int(r["size"])
                entries.append(
                    {"type": "file", "path": p, "name": name, "size": size, "depth": level}
                )
                lines.append(f"{'  ' * level}{name} ({_format_size_short(size)})")

        _walk(clean, 1)
        if truncated:
            lines.append(f"… (truncated at {cap} entries)")
        elif has_deeper:
            lines.append(f"… (deeper entries exist beyond depth {dep})")

        return {
            "path": clean,
            "depth": dep,
            "max_entries": cap,
            "truncated": truncated,
            "has_deeper": has_deeper,
            "entries": entries,
            "text": "\n".join(lines),
        }

    # ── internal write helpers ───────────────────────────────────────────────

    async def _lock(self, conn: AsyncConnection[DictRow]) -> None:
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_WRITE_LOCK_KEY,))

    async def _ensure_no_file_ancestors(
        self, conn: AsyncConnection[DictRow], paths: list[str], target: str
    ) -> None:
        if not paths:
            return
        cur = await conn.execute(
            "SELECT path FROM files WHERE path = ANY(%s) LIMIT 1",
            (paths,),
        )
        hit = await cur.fetchone()
        if hit:
            raise ConflictError(f"'{hit['path']}' is a file, so '{target}' cannot be created under it")

    async def _ensure_parents(
        self, conn: AsyncConnection[DictRow], clean_path: str, *, updated_by: str
    ) -> None:
        ancs = _ancestors(clean_path)
        if not ancs:
            return
        await self._ensure_no_file_ancestors(conn, ancs, clean_path)
        for anc in ancs:
            await conn.execute(
                """
                INSERT INTO folders (path, updated_by)
                VALUES (%s, %s)
                ON CONFLICT (path) DO NOTHING
                """,
                (anc, updated_by),
            )

    async def _check_file_collisions(self, conn: AsyncConnection[DictRow], path: str) -> None:
        """Verify `path` can be a file: no ancestor is a file, `path` is not a folder."""
        await self._ensure_no_file_ancestors(conn, _ancestors(path), path)
        cur = await conn.execute("SELECT path FROM folders WHERE path = %s LIMIT 1", (path,))
        if await cur.fetchone() is not None:
            raise ConflictError(f"'{path}' is a folder")
        cur = await conn.execute(
            "SELECT path FROM files WHERE starts_with(path, %s) LIMIT 1",
            (path + "/",),
        )
        hit = await cur.fetchone()
        if hit:
            raise ConflictError(f"'{path}' is a folder (contains '{hit['path']}')")

    # ── writes ───────────────────────────────────────────────────────────────

    async def mkdir(self, path: str, *, updated_by: str) -> dict[str, Any]:
        """Create a folder and any missing parents (idempotent `mkdir -p`)."""
        clean = normalize_entry_path(path)
        all_dirs = [*_ancestors(clean), clean]
        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute(
                "SELECT path FROM files WHERE path = ANY(%s) LIMIT 1",
                (all_dirs,),
            )
            hit = await cur.fetchone()
            if hit:
                if hit["path"] == clean:
                    raise ConflictError(f"'{clean}' already exists as a file")
                raise ConflictError(
                    f"'{hit['path']}' is a file, so folder '{clean}' cannot be created under it"
                )
            cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (clean,))
            existed = (await cur.fetchone()) is not None
            for d in all_dirs:
                await conn.execute(
                    """
                    INSERT INTO folders (path, updated_by)
                    VALUES (%s, %s)
                    ON CONFLICT (path) DO NOTHING
                    """,
                    (d, updated_by),
                )
            cur = await conn.execute(
                "SELECT path, created_at, updated_at, updated_by FROM folders WHERE path = %s",
                (clean,),
            )
            drow = await cur.fetchone()
            assert drow is not None
            out = await self._folder_stat(conn, clean, drow)
        out["created"] = not existed
        return out

    async def write(
        self,
        path: str,
        data: bytes,
        *,
        updated_by: str,
        if_sha256: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        clean = normalize_path(path)
        if len(data) > self.max_file_bytes:
            raise FileTooLargeError(
                f"file is {len(data)} bytes; the per-file limit is {self.max_file_bytes} bytes"
            )
        digest = sha256_hex(data)
        ctype = (content_type or "").strip() or guess_content_type(clean, data)
        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute("SELECT size, sha256 FROM files WHERE path = %s", (clean,))
            existing = await cur.fetchone()
            if if_sha256 is not None:
                want = if_sha256.strip().lower()
                have = str(existing["sha256"]) if existing else None
                if have != want:
                    raise ConflictError(
                        f"if_sha256 mismatch for '{clean}': "
                        + (f"current sha256 is {have}" if have else "file does not exist"),
                        current_sha256=have,
                    )
            await self._check_file_collisions(conn, clean)
            cur = await conn.execute("SELECT COALESCE(SUM(size), 0)::bigint AS used FROM files")
            used_row = await cur.fetchone()
            assert used_row is not None
            projected = int(used_row["used"]) - (int(existing["size"]) if existing else 0) + len(data)
            if projected > self.quota_bytes:
                raise QuotaExceededError(
                    f"quota exceeded: writing {len(data)} bytes would use {projected} of "
                    f"{self.quota_bytes} bytes"
                )
            await self._ensure_parents(conn, clean, updated_by=updated_by)
            cur = await conn.execute(
                f"""
                INSERT INTO files (path, content, size, sha256, content_type, updated_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (path) DO UPDATE SET
                    content = EXCLUDED.content, size = EXCLUDED.size, sha256 = EXCLUDED.sha256,
                    content_type = EXCLUDED.content_type, updated_by = EXCLUDED.updated_by,
                    updated_at = now()
                RETURNING {_META_COLS}
                """,
                (clean, data, len(data), digest, ctype, updated_by),
            )
            row = await cur.fetchone()
        assert row is not None
        out = _format_file_meta(row)
        out["created"] = existing is None
        return out

    async def delete(self, path: str, *, recursive: bool = False) -> dict[str, Any]:
        """Delete a file, an empty folder, or (with recursive=True) a folder subtree atomically."""
        clean = normalize_entry_path(path)
        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute(
                f"DELETE FROM files WHERE path = %s RETURNING {_META_COLS}",
                (clean,),
            )
            frow = await cur.fetchone()
            if frow is not None:
                out = _format_file_meta(frow)
                out["deleted_files"] = 1
                out["deleted_folders"] = 0
                return out

            cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (clean,))
            if await cur.fetchone() is None:
                raise KeyError(f"no such file or folder: {clean}")

            pfx = clean + "/"
            cur = await conn.execute(
                """
                SELECT
                    (SELECT COUNT(*)::bigint FROM files WHERE starts_with(path, %s)) AS n_files,
                    (SELECT COUNT(*)::bigint FROM folders WHERE starts_with(path, %s)) AS n_folders
                """,
                (pfx, pfx),
            )
            counts = await cur.fetchone()
            assert counts is not None
            n_files = int(counts["n_files"])
            n_folders = int(counts["n_folders"])
            if (n_files > 0 or n_folders > 0) and not recursive:
                raise ConflictError(
                    f"folder '{clean}' is not empty ({n_files} file(s), {n_folders} subfolder(s)); "
                    "pass recursive=true to delete it"
                )

            await conn.execute("DELETE FROM files WHERE starts_with(path, %s)", (pfx,))
            if self._fail_hook is not None:
                await self._fail_hook("delete_mid_subtree", conn)
            await conn.execute(
                "DELETE FROM folders WHERE path = %s OR starts_with(path, %s)",
                (clean, pfx),
            )
            return {
                "type": "folder",
                "name": clean.rsplit("/", 1)[-1],
                "path": clean,
                "deleted_files": n_files,
                "deleted_folders": n_folders + 1,
            }

    async def move(self, src: str, dst: str, *, updated_by: str) -> dict[str, Any]:
        """Move or rename a file or folder (entire subtree in one transaction). No overwrite."""
        s = normalize_entry_path(src)
        d = normalize_entry_path(dst)
        if s == d:
            raise ValueError("source and destination are the same path")

        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (s,))
            is_file = (await cur.fetchone()) is not None
            is_folder = False
            if not is_file:
                cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (s,))
                is_folder = (await cur.fetchone()) is not None
            if not is_file and not is_folder:
                raise KeyError(f"no such file or folder: {s}")

            if is_folder and d.startswith(s + "/"):
                raise ValueError(f"cannot move folder '{s}' into its own subtree '{d}'")

            # Refuse overwriting any existing file or folder at destination
            cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (d,))
            if await cur.fetchone() is not None:
                raise ConflictError(f"destination already exists: {d}")
            cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (d,))
            if await cur.fetchone() is not None:
                raise ConflictError(f"destination already exists: {d}")

            await self._ensure_no_file_ancestors(conn, _ancestors(d), d)

            if is_file:
                await self._ensure_parents(conn, d, updated_by=updated_by)
                cur = await conn.execute(
                    f"""
                    UPDATE files SET path = %s, updated_by = %s, updated_at = now()
                    WHERE path = %s RETURNING {_META_COLS}
                    """,
                    (d, updated_by, s),
                )
                row = await cur.fetchone()
                assert row is not None
                out = _format_file_meta(row)
                out["moved_files"] = 1
                out["moved_folders"] = 0
                return out

            # Folder move: check max path length across entire subtree
            s_pfx = s + "/"
            delta = len(d) - len(s)
            if delta > 0:
                cur = await conn.execute(
                    """
                    SELECT GREATEST(
                        COALESCE((SELECT MAX(char_length(path)) FROM folders WHERE path = %s OR starts_with(path, %s)), 0),
                        COALESCE((SELECT MAX(char_length(path)) FROM files WHERE starts_with(path, %s)), 0)
                    ) AS max_len
                    """,
                    (s, s_pfx, s_pfx),
                )
                ml = await cur.fetchone()
                if ml and int(ml["max_len"]) + delta > MAX_PATH_CHARS:
                    raise PathError(
                        f"moving '{s}' to '{d}' would exceed the {MAX_PATH_CHARS}-character path limit"
                    )

            await self._ensure_parents(conn, d, updated_by=updated_by)

            cur = await conn.execute(
                """
                UPDATE folders
                SET path = CASE
                    WHEN path = %(s)s THEN %(d)s
                    ELSE %(d)s || substr(path, %(s_len)s + 1)
                END,
                updated_at = now(),
                updated_by = %(who)s
                WHERE path = %(s)s OR starts_with(path, %(s_pfx)s)
                """,
                {"s": s, "d": d, "s_len": len(s), "s_pfx": s_pfx, "who": updated_by},
            )
            moved_folders = cur.rowcount

            if self._fail_hook is not None:
                await self._fail_hook("move_mid_subtree", conn)

            cur = await conn.execute(
                """
                UPDATE files
                SET path = %(d)s || substr(path, %(s_len)s + 1),
                    updated_at = now(),
                    updated_by = %(who)s
                WHERE starts_with(path, %(s_pfx)s)
                """,
                {"d": d, "s_len": len(s), "s_pfx": s_pfx, "who": updated_by},
            )
            moved_files = cur.rowcount

            cur = await conn.execute(
                "SELECT path, created_at, updated_at, updated_by FROM folders WHERE path = %s",
                (d,),
            )
            drow = await cur.fetchone()
            assert drow is not None
            out = await self._folder_stat(conn, d, drow)
            out["moved_folders"] = moved_folders
            out["moved_files"] = moved_files
            return out

    async def copy(self, src: str, dst: str, *, updated_by: str) -> dict[str, Any]:
        """Copy a file or folder recursively in one transaction, counting against quota."""
        s = normalize_entry_path(src)
        d = normalize_entry_path(dst)
        if s == d:
            raise ValueError("source and destination are the same path")

        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute("SELECT size FROM files WHERE path = %s", (s,))
            src_file = await cur.fetchone()
            is_file = src_file is not None
            is_folder = False
            if not is_file:
                cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (s,))
                is_folder = (await cur.fetchone()) is not None
            if not is_file and not is_folder:
                raise KeyError(f"no such file or folder: {s}")

            if is_folder and d.startswith(s + "/"):
                raise ValueError(f"cannot copy folder '{s}' into its own subtree '{d}'")

            cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (d,))
            if await cur.fetchone() is not None:
                raise ConflictError(f"destination already exists: {d}")
            cur = await conn.execute("SELECT 1 FROM folders WHERE path = %s", (d,))
            if await cur.fetchone() is not None:
                raise ConflictError(f"destination already exists: {d}")

            await self._ensure_no_file_ancestors(conn, _ancestors(d), d)

            cur = await conn.execute("SELECT COALESCE(SUM(size), 0)::bigint AS used FROM files")
            used_row = await cur.fetchone()
            assert used_row is not None
            used = int(used_row["used"])

            if is_file:
                copy_bytes = int(src_file["size"])
                if used + copy_bytes > self.quota_bytes:
                    raise QuotaExceededError(
                        f"quota exceeded: copying {copy_bytes} bytes would use {used + copy_bytes} "
                        f"of {self.quota_bytes} bytes"
                    )
                await self._ensure_parents(conn, d, updated_by=updated_by)
                cur = await conn.execute(
                    f"""
                    INSERT INTO files (path, content, size, sha256, content_type, updated_by)
                    SELECT %s, content, size, sha256, content_type, %s
                    FROM files WHERE path = %s
                    RETURNING {_META_COLS}
                    """,
                    (d, updated_by, s),
                )
                row = await cur.fetchone()
                assert row is not None
                out = _format_file_meta(row)
                out["copied_files"] = 1
                out["copied_folders"] = 0
                out["copied_bytes"] = copy_bytes
                return out

            # Folder copy: path length + quota check
            s_pfx = s + "/"
            delta = len(d) - len(s)
            if delta > 0:
                cur = await conn.execute(
                    """
                    SELECT GREATEST(
                        COALESCE((SELECT MAX(char_length(path)) FROM folders WHERE path = %s OR starts_with(path, %s)), 0),
                        COALESCE((SELECT MAX(char_length(path)) FROM files WHERE starts_with(path, %s)), 0)
                    ) AS max_len
                    """,
                    (s, s_pfx, s_pfx),
                )
                ml = await cur.fetchone()
                if ml and int(ml["max_len"]) + delta > MAX_PATH_CHARS:
                    raise PathError(
                        f"copying '{s}' to '{d}' would exceed the {MAX_PATH_CHARS}-character path limit"
                    )

            cur = await conn.execute(
                "SELECT COALESCE(SUM(size), 0)::bigint AS sz FROM files WHERE starts_with(path, %s)",
                (s_pfx,),
            )
            sz_row = await cur.fetchone()
            assert sz_row is not None
            copy_bytes = int(sz_row["sz"])
            if used + copy_bytes > self.quota_bytes:
                raise QuotaExceededError(
                    f"quota exceeded: copying {copy_bytes} bytes would use {used + copy_bytes} "
                    f"of {self.quota_bytes} bytes"
                )

            await self._ensure_parents(conn, d, updated_by=updated_by)

            cur = await conn.execute(
                """
                INSERT INTO folders (path, updated_by)
                SELECT
                    CASE
                        WHEN path = %(s)s THEN %(d)s
                        ELSE %(d)s || substr(path, %(s_len)s + 1)
                    END,
                    %(who)s
                FROM folders
                WHERE path = %(s)s OR starts_with(path, %(s_pfx)s)
                """,
                {"s": s, "d": d, "s_len": len(s), "s_pfx": s_pfx, "who": updated_by},
            )
            copied_folders = cur.rowcount

            if self._fail_hook is not None:
                await self._fail_hook("copy_mid_subtree", conn)

            cur = await conn.execute(
                """
                INSERT INTO files (path, content, size, sha256, content_type, updated_by)
                SELECT
                    %(d)s || substr(path, %(s_len)s + 1),
                    content, size, sha256, content_type, %(who)s
                FROM files
                WHERE starts_with(path, %(s_pfx)s)
                """,
                {"d": d, "s_len": len(s), "s_pfx": s_pfx, "who": updated_by},
            )
            copied_files = cur.rowcount

            cur = await conn.execute(
                "SELECT path, created_at, updated_at, updated_by FROM folders WHERE path = %s",
                (d,),
            )
            drow = await cur.fetchone()
            assert drow is not None
            out = await self._folder_stat(conn, d, drow)
            out["copied_folders"] = copied_folders
            out["copied_files"] = copied_files
            out["copied_bytes"] = copy_bytes
            return out
