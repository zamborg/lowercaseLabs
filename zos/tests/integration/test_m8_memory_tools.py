"""M8.4 tests for memory_list (ordering, tag filter, cursor pagination, no content) and author fields."""

from __future__ import annotations

import asyncio
import json

from zos.auth.context import CallContext
from zos.db.queries import V0_TIER_SEEDS, create_bearer_client, init_principal_and_space
from zos.policy import resolve_tier
from zos.providers.memory import MemoryProvider


async def test_memory_list_pagination_tag_filter_and_author_fields(db_pool) -> None:
    assert resolve_tier("memory_list", V0_TIER_SEEDS) == "read"

    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    c_claude = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="claude-code")
    c_codex = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="codex-cli")

    def _ctx(rec: dict) -> CallContext:
        return CallContext(
            principal_id=p_id,
            principal_slug="zubin",
            space_id=s_id,
            space_slug="main",
            client_id=rec["client"]["id"],
            client_name=rec["client"]["name"],
            client_kind="bearer",
            trust_ceiling="owner",
            token_id=rec["token_id"],
            scopes=("mcp",),
        )

    ctx_claude = _ctx(c_claude)
    ctx_codex = _ctx(c_codex)
    prov = MemoryProvider(db_pool, provider_id=init["memory_provider"]["id"])

    # Create 4 memories across two authors with distinct updated_at timestamps
    await prov.call_tool("put", {"key": "k1", "content": "Long secret body 1", "tags": ["proj:a"]}, ctx_claude)
    await asyncio.sleep(0.01)
    await prov.call_tool("put", {"key": "k2", "content": "Long secret body 2", "tags": ["proj:b"]}, ctx_codex)
    await asyncio.sleep(0.01)
    await prov.call_tool("put", {"key": "k3", "content": "Long secret body 3", "tags": ["proj:a", "pinned"]}, ctx_claude)
    await asyncio.sleep(0.01)
    await prov.call_tool("put", {"key": "k4", "content": "Long secret body 4", "tags": ["proj:a"]}, ctx_codex)

    # 1. memory_list ordered by updated_at desc, paginated with cursor, NO content field
    p1 = json.loads((await prov.call_tool("list", {"limit": 2}, ctx_claude)).content[0].text)
    assert p1["count"] == 2
    assert [i["key"] for i in p1["items"]] == ["k4", "k3"]
    assert p1["items"][0]["author"] == "codex-cli"
    assert p1["items"][1]["author"] == "claude-code"
    assert p1["next_cursor"] is not None
    for item in p1["items"]:
        assert "content" not in item
        assert "updated_at" in item
        assert "tags" in item

    p2 = json.loads((await prov.call_tool("list", {"limit": 2, "cursor": p1["next_cursor"]}, ctx_claude)).content[0].text)
    assert p2["count"] == 2
    assert [i["key"] for i in p2["items"]] == ["k2", "k1"]
    assert p2["items"][0]["author"] == "codex-cli"
    assert p2["items"][1]["author"] == "claude-code"
    assert p2["next_cursor"] is None

    # 2. Tag filter
    p_tag = json.loads((await prov.call_tool("list", {"tag": "proj:a", "limit": 10}, ctx_claude)).content[0].text)
    assert [i["key"] for i in p_tag["items"]] == ["k4", "k3", "k1"]

    # 3. Invalid cursor -> clean error
    bad_cur = await prov.call_tool("list", {"cursor": "not-valid-base64!!!"}, ctx_claude)
    assert bad_cur.is_error is True

    # 4. memory_get & memory_search return created_at, updated_at, and author
    got = json.loads((await prov.call_tool("get", {"key": "k4"}, ctx_claude)).content[0].text)
    assert got["found"] is True
    assert got["memory"]["author"] == "codex-cli"
    assert got["memory"]["created_at"]
    assert got["memory"]["updated_at"]

    srch = json.loads((await prov.call_tool("search", {"query": "secret body 3"}, ctx_codex)).content[0].text)
    assert srch["count"] >= 1
    hit = next(m for m in srch["matches"] if m["key"] == "k3")
    assert hit["author"] == "claude-code"
    assert hit["created_at"]
    assert hit["updated_at"]
