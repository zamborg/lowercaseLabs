"""Integration tests for M1.7 append-only tool call auditing."""

from __future__ import annotations

import hashlib
import json

from zos.audit import DatabaseAuditRecorder, args_digest, canonical_json
from zos.auth import verify_client_bearer_token
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.router import ProviderRegistry, ToolRouter


def test_args_digest_is_canonical_and_order_independent() -> None:
    d1 = args_digest({"b": 2, "a": [1, {"y": True, "x": "secret"}]})
    d2 = args_digest({"a": [1, {"x": "secret", "y": True}], "b": 2})
    assert d1 == d2
    assert d1 == hashlib.sha256(canonical_json({"b": 2, "a": [1, {"y": True, "x": "secret"}]}).encode()).digest()


async def test_router_records_tool_called_event_without_raw_args(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    created = await create_bearer_client(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        name="audit-client",
    )
    ctx = await verify_client_bearer_token(db_pool, created["token"])
    assert ctx is not None

    audit = DatabaseAuditRecorder(db_pool)
    router = ToolRouter(ProviderRegistry(), record_audit=audit.record_tool_call)

    secret_val = "SUPER_PRIVATE_ARG_CONTENT_999"
    await router.call_tool("zos_status", {}, ctx)
    await router.call_tool("nonexistent_tool", {"secret": secret_val}, ctx)

    async with db_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, client_id, provider_id, type, tool, decision, outcome, args_digest, payload "
            "FROM events WHERE type = 'tool.called' ORDER BY id ASC"
        )
        rows = await cur.fetchall()

    assert len(rows) == 2
    assert rows[0]["tool"] == "zos_status"
    assert rows[0]["outcome"] == "ok"
    assert rows[0]["client_id"] == ctx.client_id
    assert bytes(rows[0]["args_digest"]) == args_digest({})

    assert rows[1]["tool"] == "nonexistent_tool"
    assert rows[1]["outcome"] == "error"
    assert bytes(rows[1]["args_digest"]) == args_digest({"secret": secret_val})
    assert secret_val not in json.dumps(rows[1]["payload"])
    assert rows[1]["id"] > rows[0]["id"]
