"""Integration tests for M12a: zero-module core startup, memory manifest, and data migration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx2
import psycopg
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import run_asgi_server
from zos.app import create_app
from zos.catalog import get_module_catalog_entry, load_module_catalog
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.modules.storage import ModuleStorageProvisioner
from zos.vault import CredentialStore, CredentialVault, generate_secret_key


def _load_migrate_script():
    script_path = (
        Path(__file__).resolve().parents[2] / "deploy" / "scripts" / "migrate-memory-to-module.py"
    )
    spec = importlib.util.spec_from_file_location("migrate_memory_to_module", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checked_in_memory_module_manifest_is_valid() -> None:
    entries = load_module_catalog()
    assert any(e.slug == "memory" for e in entries)

    mem = get_module_catalog_entry("memory")
    assert mem is not None
    assert mem.manifest_version == 2
    assert mem.runtime.kind == "container"
    assert mem.runtime.build == "modules/memory"
    assert mem.runtime.mcp_port == 8081
    assert mem.runtime.mcp_path == "/mcp"
    assert mem.runtime.ui_port == 8082
    assert mem.storage.postgres is True
    assert mem.storage.extensions == ["pg_trgm"]
    assert mem.network.egress is False
    assert mem.ui is not None and mem.ui.title == "Memory"
    assert mem.tiers.patterns["memory_get"] == "read"
    assert mem.tiers.patterns["memory_search"] == "read"
    assert mem.tiers.patterns["memory_list"] == "read"
    assert mem.tiers.patterns["memory_put"] == "write"
    assert mem.tiers.patterns["memory_delete"] == "write"

    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "modules" / "memory" / "Dockerfile").is_file()
    assert (repo_root / "modules" / "memory" / "pyproject.toml").is_file()


async def test_core_starts_and_routes_with_zero_modules_installed(db_pool) -> None:
    """Core starts cleanly and routes zos_status over MCP with zero modules installed."""
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    client = await create_bearer_client(
        db_pool,
        principal_id=init["principal"]["id"],
        space_id=init["space"]["id"],
        name="zero-mod-client",
    )

    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://127.0.0.1:0",
    )
    app = create_app(cfg, run_migrations_on_startup=False)

    async with run_asgi_server(app) as base_url:
        http_c = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client['token']}"})
        async with (
            http_c,
            streamable_http_client(f"{base_url}/mcp", http_client=http_c) as (r, w),
            ClientSession(r, w) as sess,
        ):
            await sess.initialize()
            listed = await sess.list_tools()
            names = {t.name for t in listed.tools}
            assert "zos_status" in names
            assert not any(n.startswith("memory_") for n in names)

            res = await sess.call_tool("zos_status", {})
            assert not res.is_error
            payload = json.loads(res.content[0].text)
            assert "providers" in payload
            assert not any(p["namespace"] == "memory" for p in payload["providers"])


async def test_migrate_memory_script_copies_with_checksum_and_is_idempotent(db_pool) -> None:
    """Data migration copies public.memories -> mod_memory.memories idempotently with SHA-256 verification."""
    db_url = get_test_database_url()
    init = await init_principal_and_space(db_pool, principal_slug="owner")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]
    client = await create_bearer_client(
        db_pool,
        principal_id=p_id,
        space_id=s_id,
        name="claude-migrator",
    )

    async with db_pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO public.memories (principal_id, space_id, key, content, tags, created_by_client_id)
            VALUES
              (%s, %s, 'pref/tea', 'Prefers sencha green tea', ARRAY['prefs', 'tea'], %s),
              (%s, %s, 'proj/zos', 'Core is control plane only', ARRAY['work'], NULL)
            """,
            (p_id, s_id, client["client"]["id"], p_id, s_id),
        )
        await conn.commit()

    store = CredentialStore(db_pool, CredentialVault(generate_secret_key()))
    prov = ModuleStorageProvisioner(db_pool, store)
    mod_dsn = await prov.provision_postgres(
        principal_id=p_id,
        slug="memory",
        base_database_url=db_url,
        extensions=["pg_trgm"],
    )
    try:
        # Initialize module table as the module role
        async with await psycopg.AsyncConnection.connect(mod_dsn, autocommit=True) as mod_conn:
            await mod_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    tags TEXT[] NOT NULL DEFAULT '{}',
                    fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', key || ' ' || content)) STORED,
                    created_by_client_id TEXT NULL,
                    created_by_client_name TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS memories_content_trgm_gin_idx ON memories USING GIN (content gin_trgm_ops);
                """
            )

        mod = _load_migrate_script()
        res1 = mod.migrate_memories(db_url)
        assert res1["ok"] is True
        assert res1["source_count"] == 2
        assert res1["target_count"] == 2
        assert res1["inserted"] == 2
        assert res1["skipped"] == 0
        assert len(res1["sha256"]) == 64

        # Second run is strictly idempotent with identical checksum
        res2 = mod.migrate_memories(db_url)
        assert res2["ok"] is True
        assert res2["inserted"] == 0
        assert res2["skipped"] == 2
        assert res2["sha256"] == res1["sha256"]

        # Verify creator name survived and legacy public.memories is still intact
        async with db_pool.connection() as conn:
            m_row = await (
                await conn.execute(
                    "SELECT created_by_client_name FROM mod_memory.memories WHERE key = 'pref/tea'"
                )
            ).fetchone()
            assert m_row is not None and m_row["created_by_client_name"] == "claude-migrator"
            legacy_cnt = await (
                await conn.execute("SELECT COUNT(*) AS cnt FROM public.memories")
            ).fetchone()
            assert legacy_cnt is not None and legacy_cnt["cnt"] == 2
    finally:
        await prov.purge_storage(principal_id=p_id, slug="memory")
