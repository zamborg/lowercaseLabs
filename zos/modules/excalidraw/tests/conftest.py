import os
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo

from zos_excalidraw.caller import sign_http_caller
from zos_excalidraw.store import DrawingStore, create_pool


@pytest.fixture
async def store():
    url = os.environ.get("TEST_DATABASE_URL", "postgresql://zos:zos@127.0.0.1:55432/zos_test")
    schema = "excalidraw_test_" + uuid4().hex
    async with await AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            pool = create_pool(make_conninfo(url, options=f"-csearch_path={schema}"))
            async with pool:
                drawings = DrawingStore(pool)
                await drawings.init_schema()
                yield drawings
        finally:
            await conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def owner_headers():
    header, signature = sign_http_caller("fixture", principal_id="test-owner")
    return {"X-ZOS-Caller": header, "X-ZOS-Caller-Signature": signature}
