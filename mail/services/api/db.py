from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from services.api.settings import Settings


class Database:
    def __init__(self, settings: Settings) -> None:
        self._pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
        )
        self._schema_path = Path(__file__).with_name("schema.sql")

    @property
    def pool(self) -> ConnectionPool:
        return self._pool

    def close(self) -> None:
        self._pool.close()

    def init_schema(self) -> None:
        schema_sql = self._schema_path.read_text(encoding="utf-8")
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)

    def fetch_one(self, query: str, params: Optional[Iterable[Any]] = None) -> Optional[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.fetchone()

    def fetch_all(self, query: str, params: Optional[Iterable[Any]] = None) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return list(cur.fetchall())

    def execute(self, query: str, params: Optional[Iterable[Any]] = None) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                conn.commit()
