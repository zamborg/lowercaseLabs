"""Integration tests for M1.8 Postgres-backed MemoryProvider."""

from __future__ import annotations

import json

from zos.auth import verify_client_bearer_token
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.providers.memory import MemoryProvider


async def test_memory_put_get_search_trigram_tags_limit_and_delete(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    created = await create_bearer_client(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        name="mem-client",
    )
    ctx = await verify_client_bearer_token(db_pool, created["token"])
    assert ctx is not None

    provider = MemoryProvider(db_pool, provider_id=init["memory_provider"]["id"])

    # Verify destructive hint on delete only
    tools = {t.name: t for t in await provider.list_tools()}
    assert tools["delete"].destructive_hint is True
    assert tools["put"].destructive_hint is False

    # Put + upsert
    r1 = await provider.call_tool(
        "put",
        {"key": "coffee-pref", "content": "Zubin prefers single-origin pour-over espresso.", "tags": ["food", "prefs"]},
        ctx,
    )
    assert not r1.is_error
    r1_upd = await provider.call_tool(
        "put",
        {"key": "coffee-pref", "content": "Zubin prefers oat flat white and cortado.", "tags": ["food", "prefs"]},
        ctx,
    )
    assert not r1_upd.is_error

    await provider.call_tool(
        "put",
        {"key": "project-zos", "content": "ZOS routes MCP tool calls with Postgres memory.", "tags": ["work", "zos"]},
        ctx,
    )

    # Get
    got = json.loads((await provider.call_tool("get", {"key": "coffee-pref"}, ctx)).content[0].text)
    assert got["found"] is True
    assert "cortado" in got["memory"]["content"]

    missing = json.loads((await provider.call_tool("get", {"key": "does-not-exist"}, ctx)).content[0].text)
    assert missing["found"] is False

    # FTS search
    fts = json.loads((await provider.call_tool("search", {"query": "Postgres memory"}, ctx)).content[0].text)
    assert fts["mode"] == "fts"
    assert fts["count"] == 1
    assert fts["matches"][0]["key"] == "project-zos"

    # Trigram fallback search (typo: 'cortadoo')
    trgm = json.loads((await provider.call_tool("search", {"query": "cortadoo"}, ctx)).content[0].text)
    assert trgm["mode"] == "trigram"
    assert trgm["count"] == 1
    assert trgm["matches"][0]["key"] == "coffee-pref"

    # Tag filter
    tagged = json.loads((await provider.call_tool("search", {"tags": ["work"]}, ctx)).content[0].text)
    assert [m["key"] for m in tagged["matches"]] == ["project-zos"]

    # Limit clamped <= 50
    for i in range(55):
        await provider.call_tool("put", {"key": f"bulk-{i:02d}", "content": f"bulk item {i}", "tags": ["bulk"]}, ctx)
    bulk = json.loads((await provider.call_tool("search", {"tags": ["bulk"], "limit": 999}, ctx)).content[0].text)
    assert bulk["count"] == 50

    # Delete
    d1 = json.loads((await provider.call_tool("delete", {"key": "coffee-pref"}, ctx)).content[0].text)
    assert d1["deleted"] is True
    d2 = json.loads((await provider.call_tool("delete", {"key": "coffee-pref"}, ctx)).content[0].text)
    assert d2["deleted"] is False
