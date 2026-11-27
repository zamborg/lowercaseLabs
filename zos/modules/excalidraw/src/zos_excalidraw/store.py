from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .scenes import convert, edit_scene, normalize_scene, summary


class Conflict(Exception):
    def __init__(self, row):
        self.detail = {"error": "Version conflict", "current_version": row["version"], "summary": summary(row["scene"])}
        super().__init__("Version conflict")


def create_pool(url):
    return AsyncConnectionPool(url, open=False, kwargs={"row_factory": dict_row})


def clean_title(title):
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 500:
        raise ValueError("Title must be 1 to 500 characters")
    return title.strip()


def serialize(row):
    return {**row, "id": str(row["id"]), "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat()}


class DrawingStore:
    def __init__(self, pool):
        self.pool = pool

    async def init_schema(self):
        async with self.pool.connection() as conn:
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mod_excalidraw.migrations'))")
            await conn.execute("CREATE TABLE IF NOT EXISTS drawing_migrations (version integer PRIMARY KEY)")
            exists = await (await conn.execute("SELECT version FROM drawing_migrations WHERE version=1")).fetchone()
            if not exists:
                await conn.execute("""CREATE TABLE drawings (
                    id uuid PRIMARY KEY, title text NOT NULL, scene jsonb NOT NULL, updated_by text NOT NULL,
                    version bigint NOT NULL DEFAULT 1, created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now())""")
                await conn.execute("INSERT INTO drawing_migrations VALUES (1)")

    async def list(self):
        async with self.pool.connection() as conn:
            rows = await (await conn.execute("""SELECT id,title,updated_by,version,updated_at,created_at
                FROM drawings ORDER BY updated_at DESC,id""")).fetchall()
            return [serialize(r) for r in rows]

    async def get(self, id):
        async with self.pool.connection() as conn:
            row = await (await conn.execute("SELECT * FROM drawings WHERE id=%s", (UUID(id),))).fetchone()
            if not row:
                raise KeyError("Drawing not found")
            return {**serialize(row), "summary": summary(row["scene"])}

    async def create(self, title, actor, elements):
        scene = normalize_scene({"elements": convert(elements)})
        async with self.pool.connection() as conn:
            row = await (await conn.execute("""INSERT INTO drawings (id,title,scene,updated_by)
                VALUES (%s,%s,%s,%s) RETURNING *""", (uuid4(), clean_title(title), Jsonb(scene), actor))).fetchone()
            return serialize(row)

    async def change(self, id, version, actor, *, action, data):
        if type(version) is not int or version < 1:
            raise ValueError("if_version must be a positive integer")
        async with self.pool.connection() as conn:
            row = await (await conn.execute("SELECT * FROM drawings WHERE id=%s FOR UPDATE", (UUID(id),))).fetchone()
            if not row:
                raise KeyError("Drawing not found")
            if row["version"] != version:
                raise Conflict(row)
            if action == "delete":
                await conn.execute("DELETE FROM drawings WHERE id=%s", (row["id"],))
                return {"deleted": True}
            title, scene = row["title"], row["scene"]
            if action == "rename":
                title = clean_title(data)
            elif action == "save":
                scene = normalize_scene(data)
            elif action == "update":
                scene = edit_scene(scene, data)
            else:
                raise ValueError("Unknown action")
            updated = await (await conn.execute("""UPDATE drawings SET scene=%s, title=%s, updated_by=%s,
                version=version+1, updated_at=clock_timestamp() WHERE id=%s RETURNING version""",
                (Jsonb(scene), title, actor, row["id"]))).fetchone()
            return {"id": id, **updated}
