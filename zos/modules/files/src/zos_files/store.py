"""Postgres storage layer for the ZOS Files module (uses only its own schema).

Files live in a single table keyed by a normalized, unique, `/`-separated path.
Folders do not exist as rows: they are just path prefixes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
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

# Serializes all writers so quota and `if_sha256` checks are race-free.
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
]


class PathError(ValueError):
    """The path (or prefix) violates the path rules."""


class FileTooLargeError(ValueError):
    """A single file exceeds the per-file size limit."""


class QuotaExceededError(ValueError):
    """The write would push total usage past the configured quota."""


class ConflictError(Exception):
    """Optimistic-concurrency or path-collision conflict."""

    def __init__(self, message: str, *, current_sha256: str | None = None) -> None:
        super().__init__(message)
        self.current_sha256 = current_sha256


def normalize_path(raw: str) -> str:
    """Validate and return a canonical file path (relative, `/`-separated, no `..`)."""
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


def normalize_prefix(raw: str | None) -> str:
    """Normalize a folder prefix to either '' (root) or 'a/b/' (trailing slash)."""
    if raw is None:
        return ""
    trimmed = raw.strip()
    if trimmed in ("", "/"):
        return ""
    if trimmed.endswith("/"):
        trimmed = trimmed[:-1]
    return normalize_path(trimmed) + "/"


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


def _format_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(row["path"]),
        "size": int(row["size"]),
        "sha256": str(row["sha256"]),
        "content_type": str(row["content_type"]),
        "updated_at": _iso(row["updated_at"]),
        "updated_by": str(row["updated_by"]),
    }


_META_COLS = "path, size, sha256, content_type, updated_at, updated_by"


def create_files_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


class FilesStore:
    """Async Postgres store for path-addressed files."""

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
                "SELECT COALESCE(SUM(size), 0)::bigint AS used, COUNT(*)::bigint AS n FROM files"
            )
            row = await cur.fetchone()
        assert row is not None
        return {
            "used_bytes": int(row["used"]),
            "file_count": int(row["n"]),
            "quota_bytes": self.quota_bytes,
            "max_file_bytes": self.max_file_bytes,
        }

    async def stat(self, path: str) -> dict[str, Any]:
        clean = normalize_path(path)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_META_COLS} FROM files WHERE path = %s",
                (clean,),
            )
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"no such file: {clean}")
        return _format_meta(row)

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
        return _format_meta(row), bytes(row["content"])

    async def list(
        self,
        *,
        prefix: str | None = None,
        recursive: bool = True,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        pfx = normalize_prefix(prefix)
        lim = int(limit)
        if lim < 1 or lim > MAX_LIST_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
        after = cursor or ""

        async with self.pool.connection() as conn:
            if recursive:
                cur = await conn.execute(
                    f"""
                    SELECT {_META_COLS} FROM files
                    WHERE starts_with(path, %s) AND path COLLATE "C" > %s
                    ORDER BY path COLLATE "C"
                    LIMIT %s
                    """,
                    (pfx, after, lim + 1),
                )
                rows = await cur.fetchall()
                page = rows[:lim]
                entries = [{"type": "file", **_format_meta(r)} for r in page]
                next_cursor = str(page[-1]["path"]) if len(rows) > lim and page else None
            else:
                cur = await conn.execute(
                    """
                    WITH r AS (
                        SELECT path, substr(path, %(n)s + 1) AS rel, size, sha256,
                               content_type, updated_at, updated_by
                        FROM files WHERE starts_with(path, %(pfx)s)
                    ), g AS (
                        SELECT split_part(rel, '/', 1) AS name,
                               position('/' in rel) > 0 AS is_dir,
                               COUNT(*)::bigint AS file_count,
                               SUM(size)::bigint AS size,
                               MAX(updated_at) AS updated_at,
                               MIN(sha256) AS sha256,
                               MIN(content_type) AS content_type,
                               MIN(updated_by) AS updated_by
                        FROM r GROUP BY 1, 2
                    )
                    SELECT * FROM g WHERE name COLLATE "C" > %(after)s
                    ORDER BY name COLLATE "C" LIMIT %(lim)s
                    """,
                    {"n": len(pfx), "pfx": pfx, "after": after, "lim": lim + 1},
                )
                rows = await cur.fetchall()
                page = rows[:lim]
                entries = []
                for r in page:
                    if r["is_dir"]:
                        entries.append(
                            {
                                "type": "folder",
                                "path": f"{pfx}{r['name']}/",
                                "file_count": int(r["file_count"]),
                                "size": int(r["size"]),
                                "updated_at": _iso(r["updated_at"]),
                            }
                        )
                    else:
                        entries.append(
                            {
                                "type": "file",
                                **_format_meta({**r, "path": f"{pfx}{r['name']}"}),
                            }
                        )
                next_cursor = str(page[-1]["name"]) if len(rows) > lim and page else None
        return {"prefix": pfx, "recursive": recursive, "entries": entries, "next_cursor": next_cursor}

    # ── writes ───────────────────────────────────────────────────────────────

    async def _lock(self, conn: AsyncConnection[DictRow]) -> None:
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_WRITE_LOCK_KEY,))

    async def _check_collisions(
        self, conn: AsyncConnection[DictRow], path: str, *, ignore: str | None = None
    ) -> None:
        """A path can't be both a file and a folder."""
        segs = path.split("/")
        ancestors = ["/".join(segs[:i]) for i in range(1, len(segs))]
        if ancestors:
            cur = await conn.execute(
                "SELECT path FROM files WHERE path = ANY(%s) AND path IS DISTINCT FROM %s LIMIT 1",
                (ancestors, ignore),
            )
            hit = await cur.fetchone()
            if hit:
                raise ConflictError(f"'{hit['path']}' is a file, so '{path}' cannot be created under it")
        cur = await conn.execute(
            "SELECT path FROM files WHERE starts_with(path, %s) AND path IS DISTINCT FROM %s LIMIT 1",
            (path + "/", ignore),
        )
        hit = await cur.fetchone()
        if hit:
            raise ConflictError(f"'{path}' is a folder (contains '{hit['path']}')")

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
            await self._check_collisions(conn, clean)
            cur = await conn.execute("SELECT COALESCE(SUM(size), 0)::bigint AS used FROM files")
            used_row = await cur.fetchone()
            assert used_row is not None
            projected = int(used_row["used"]) - (int(existing["size"]) if existing else 0) + len(data)
            if projected > self.quota_bytes:
                raise QuotaExceededError(
                    f"quota exceeded: writing {len(data)} bytes would use {projected} of "
                    f"{self.quota_bytes} bytes"
                )
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
        out = _format_meta(row)
        out["created"] = existing is None
        return out

    async def delete(self, path: str) -> dict[str, Any]:
        clean = normalize_path(path)
        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute(
                f"DELETE FROM files WHERE path = %s RETURNING {_META_COLS}",
                (clean,),
            )
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"no such file: {clean}")
        return _format_meta(row)

    async def move(self, src: str, dst: str, *, updated_by: str) -> dict[str, Any]:
        s = normalize_path(src)
        d = normalize_path(dst)
        if s == d:
            raise ValueError("source and destination are the same path")
        async with self.pool.connection() as conn, conn.transaction():
            await self._lock(conn)
            cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (s,))
            if await cur.fetchone() is None:
                raise KeyError(f"no such file: {s}")
            cur = await conn.execute("SELECT 1 FROM files WHERE path = %s", (d,))
            if await cur.fetchone() is not None:
                raise ConflictError(f"destination already exists: {d}")
            await self._check_collisions(conn, d, ignore=s)
            cur = await conn.execute(
                f"""
                UPDATE files SET path = %s, updated_by = %s, updated_at = now()
                WHERE path = %s RETURNING {_META_COLS}
                """,
                (d, updated_by, s),
            )
            row = await cur.fetchone()
        assert row is not None
        return _format_meta(row)
