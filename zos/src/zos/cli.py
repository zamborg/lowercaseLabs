"""ZOS command-line interface (`zos`).

Delegates all administrative mutations to the shared `AdminService` layer
so CLI and `/admin` share identical behavior and audit events.
"""

from __future__ import annotations

import asyncio
import getpass
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer
from argon2 import PasswordHasher

from zos.admin_service import AdminService
from zos.catalog import load_catalog, load_module_catalog
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import init_principal_and_space
from zos.importers import IMPORTERS, dumps_jsonl, loads_jsonl
from zos.memory_service import MemoryService
from zos.modules import ModuleManager

app = typer.Typer(name="zos", help="ZOS MCP router administration CLI.", no_args_is_help=True)
client_app = typer.Typer(help="Manage MCP clients.")
token_app = typer.Typer(help="Manage bearer and OAuth tokens.")
policy_app = typer.Typer(help="Manage per-client and default tool policies.")
actuator_app = typer.Typer(help="Manage dial-out actuators.")
provider_app = typer.Typer(help="Manage upstream providers.")
upstream_app = typer.Typer(help="Manage routed remote MCP servers.")
catalog_app = typer.Typer(help="Browse and install connectors from the built-in catalog.")
module_app = typer.Typer(help="Manage installable ZOS container modules.")
memory_app = typer.Typer(help="Import and export agent memories.")
backup_app = typer.Typer(help="Record encrypted backup status.")

app.add_typer(client_app, name="client")
app.add_typer(token_app, name="token")
app.add_typer(policy_app, name="policy")
app.add_typer(actuator_app, name="actuator")
app.add_typer(provider_app, name="provider")
app.add_typer(upstream_app, name="upstream")
app.add_typer(catalog_app, name="catalog")
app.add_typer(module_app, name="module")
app.add_typer(memory_app, name="memory")
app.add_typer(backup_app, name="backup")


def _settings() -> Settings:
    return Settings.from_env()


@app.command("migrate")
def migrate_cmd() -> None:
    """Apply pending SQL migrations."""

    async def _run() -> list[str]:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await run_migrations(pool, s.migrations_dir)

    applied = asyncio.run(_run())
    if applied:
        typer.echo(f"Applied migrations: {', '.join(applied)}")
    else:
        typer.echo("Database is up to date.")


@app.command("init")
def init_cmd(
    principal: Annotated[str, typer.Option("--principal", help="Principal slug")] = "owner",
    display_name: Annotated[str, typer.Option("--display-name", help="Principal display name")] = "Owner",
    space: Annotated[str, typer.Option("--space", help="Default space slug")] = "main",
    space_name: Annotated[str, typer.Option("--space-name", help="Default space name")] = "Main",
) -> None:
    """Initialize principal, single space, memory provider, and V0 tool tier seeds."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            await run_migrations(pool, s.migrations_dir)
            return await init_principal_and_space(
                pool,
                principal_slug=principal,
                display_name=display_name,
                space_slug=space,
                space_name=space_name,
            )

    res = asyncio.run(_run())
    typer.echo(
        f"Initialized principal={res['principal']['slug']} "
        f"space={res['space']['slug']} "
        f"provider={res['memory_provider']['slug']} "
        f"tiers={res['tier_seed_count']}"
    )


@client_app.command("add")
def client_add_cmd(
    name: Annotated[str, typer.Argument(help="Client name")],
    bearer: Annotated[bool, typer.Option("--bearer", help="Create a bearer-token client")] = False,
    trust_ceiling: Annotated[str, typer.Option("--trust-ceiling", help="owner or untrusted")] = "owner",
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Register a new bearer client and print its token once."""
    if not bearer:
        typer.echo("Error: pass --bearer to register a bearer client (OAuth clients register via DCR).", err=True)
        raise typer.Exit(code=2)
    if trust_ceiling not in ("owner", "untrusted"):
        typer.echo("Error: --trust-ceiling must be 'owner' or 'untrusted'.", err=True)
        raise typer.Exit(code=2)

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).register_bearer_client(
                name=name,
                trust_ceiling=trust_ceiling,
                principal_slug=principal,
            )

    res = asyncio.run(_run())
    typer.echo(f"client_id={res['client']['id']}")
    typer.echo(f"token_id={res['token_id']}")
    typer.echo(f"token={res['token']}")


@client_app.command("delete")
def client_delete_cmd(
    client: Annotated[str, typer.Argument(help="OAuth client UUID, oauth_client_id, or name")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Prune an OAuth client (cascades tokens/codes/policies, preserves audit events)."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).delete_oauth_client(
                client_identifier=client,
                principal_slug=principal,
            )

    res = asyncio.run(_run())
    typer.echo(f"Pruned OAuth client name={res['name']} id={res['id']}")


@client_app.command("disable")
def client_disable_cmd(
    client: Annotated[str, typer.Argument(help="Client UUID, oauth_client_id, or name")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Soft-disable any client and revoke its tokens while preserving audit history and memory authorship."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).disable_client(
                client_identifier=client,
                principal_slug=principal,
            )

    res = asyncio.run(_run())
    typer.echo(
        f"Disabled client name={res['name']} id={res['id']} revoked_tokens={res['revoked_tokens']}"
    )


@token_app.command("create")
def token_create_cmd(
    client_name: Annotated[str, typer.Argument(help="Existing bearer client name")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Mint an additional bearer token for an existing bearer client."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).create_bearer_token(
                client_name=client_name,
                principal_slug=principal,
            )

    res = asyncio.run(_run())
    typer.echo(f"token_id={res['token_id']}")
    typer.echo(f"token={res['token']}")


@token_app.command("revoke")
def token_revoke_cmd(
    target: Annotated[str | None, typer.Argument(help="Token UUID or raw token string")] = None,
    client: Annotated[str | None, typer.Option("--client", help="Revoke all tokens for client name/id")] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Revoke a token by UUID/raw value or revoke all tokens for a client."""
    if not target and not client:
        typer.echo("Error: pass a token ID/value or --client NAME.", err=True)
        raise typer.Exit(code=2)

    async def _run() -> int:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).revoke_tokens(
                target=target,
                client_identifier=client,
                principal_slug=principal,
            )

    count = asyncio.run(_run())
    typer.echo(f"Revoked {count} token(s).")


@policy_app.command("set")
def policy_set_cmd(
    pattern: Annotated[str, typer.Argument(help="Tool name or pattern (e.g. gmail_send or mac_*)")],
    decision: Annotated[str, typer.Argument(help="allow | deny | require_approval")],
    client: Annotated[str | None, typer.Option("--client", help="Client name or ID (omit for all clients)")] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Set a policy decision for a tool pattern globally or for a specific client."""
    if decision not in ("allow", "deny", "require_approval"):
        typer.echo("Error: DECISION must be one of: allow, deny, require_approval.", err=True)
        raise typer.Exit(code=2)

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).set_policy(
                pattern=pattern,
                decision=decision,
                client_identifier=client,
                principal_slug=principal,
            )

    row = asyncio.run(_run())
    scope_label = client or "*"
    typer.echo(f"Policy set: client={scope_label} pattern={row['tool_pattern']} decision={row['decision']}")


@policy_app.command("clear")
def policy_clear_cmd(
    pattern: Annotated[str, typer.Argument(help="Tool name or pattern to clear override for")],
    client: Annotated[str | None, typer.Option("--client", help="Client name or ID (omit for all clients)")] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Clear a policy override so resolution falls back to global/tier default."""

    async def _run() -> bool:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).clear_policy(
                pattern=pattern,
                client_identifier=client,
                principal_slug=principal,
            )

    deleted = asyncio.run(_run())
    scope_label = client or "*"
    if deleted:
        typer.echo(f"Policy cleared: client={scope_label} pattern={pattern}")
    else:
        typer.echo(f"No policy override found: client={scope_label} pattern={pattern}")


@actuator_app.command("add")
def actuator_add_cmd(
    slug: Annotated[str, typer.Argument(help="Actuator provider slug (e.g. mac)")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Register an actuator provider and print its actuator token once."""
    if "_" in slug or not slug.replace("-", "").isalnum():
        typer.echo("Error: actuator slug must match [a-zA-Z0-9-]+ with no underscores.", err=True)
        raise typer.Exit(code=2)

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).add_actuator(
                slug=slug,
                principal_slug=principal,
            )

    res = asyncio.run(_run())
    typer.echo(f"provider_id={res['provider']['id']}")
    typer.echo(f"token_id={res['token_id']}")
    typer.echo(f"token={res['token']}")


@provider_app.command("set-backend")
def provider_set_backend_cmd(
    slug: Annotated[str, typer.Argument(help="Upstream provider slug (e.g. gcal)")],
    backend: Annotated[str, typer.Argument(help="Backend: api | mcp")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Switch an upstream provider's backend (persisted & audited)."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).set_provider_backend(
                slug=slug,
                backend=backend,
                principal_slug=principal,
            )

    try:
        res = asyncio.run(_run())
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    prev = res.get("previous_backend") or "default"
    typer.echo(f"Provider backend set: slug={res['slug']} from={prev} to={res['backend']}")


@upstream_app.command("add")
def upstream_add_cmd(
    slug: Annotated[str, typer.Argument(help="Unique tool prefix (^[a-z][a-z0-9]{1,15}$, not 'zos')")],
    url: Annotated[str, typer.Option("--url", help="Remote Streamable HTTP MCP server URL")],
    auth: Annotated[str, typer.Option("--auth", help="Auth mode: none | bearer | oauth")] = "none",
    default_tier: Annotated[
        str,
        typer.Option("--default-tier", help="Default tier for tools from this server: read | write | external"),
    ] = "external",
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Register a remote MCP server (`runtime: remote`)."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).add_remote_upstream(
                slug=slug,
                url=url,
                auth_mode=auth,
                default_tier=default_tier,
                principal_slug=principal,
            )

    try:
        row = asyncio.run(_run())
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    cfg = row["config"]
    typer.echo(
        f"Added upstream slug={row['slug']} url={cfg['url']} "
        f"auth={cfg['auth_mode']} default_tier={cfg['default_tier']}"
    )


@upstream_app.command("set-token")
def upstream_set_token_cmd(
    slug: Annotated[str, typer.Argument(help="Remote MCP upstream slug (auth_mode=bearer)")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Read a bearer token/PAT strictly from stdin and store it encrypted in the vault."""
    if sys.stdin.isatty():
        token = getpass.getpass(f"Bearer token for '{slug}': ").strip()
    else:
        token = sys.stdin.read().strip()
    if not token:
        typer.echo("Error: empty token provided on stdin.", err=True)
        raise typer.Exit(code=2)

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).set_upstream_bearer_token(
                slug=slug,
                token=token,
                secret_key=s.zos_secret_key,
                principal_slug=principal,
            )

    try:
        res = asyncio.run(_run())
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Stored encrypted bearer token for slug={res['slug']}")


@upstream_app.command("connect")
def upstream_connect_cmd(
    slug: Annotated[str, typer.Argument(help="Remote MCP upstream slug (auth_mode=oauth)")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Print the owner-gated browser /connect/<slug> URL for an OAuth remote MCP server."""

    async def _run() -> tuple[str, dict | None]:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            items = await AdminService(pool).list_remote_upstreams(principal_slug=principal)
        match = next((i for i in items if i["slug"] == slug.strip().lower()), None)
        return s.zos_base_url.rstrip("/"), match

    base_url, match = asyncio.run(_run())
    if match is None:
        typer.echo(f"Error: remote MCP upstream not found: {slug}", err=True)
        raise typer.Exit(code=2)
    if match["auth_mode"] != "oauth":
        typer.echo(
            f"Error: upstream '{match['slug']}' uses auth_mode='{match['auth_mode']}', not 'oauth'.",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(f"{base_url}/connect/{match['slug']}")


@upstream_app.command("list")
def upstream_list_cmd(
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """List registered remote MCP servers with status and tool count."""

    async def _run() -> list[dict]:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            items = await AdminService(pool).list_remote_upstreams(principal_slug=principal)
        # Best-effort merge of live status & tool_count from local running ZOS /healthz
        live_by_ns: dict[str, dict] = {}
        for probe_url in ("http://127.0.0.1:8080/healthz", f"{s.zos_base_url.rstrip('/')}/healthz"):
            try:
                async with httpx.AsyncClient(timeout=1.5) as c:
                    r = await c.get(probe_url)
                    if r.status_code == 200:
                        for p in r.json().get("providers", []):
                            live_by_ns[str(p["namespace"])] = p
                        break
            except Exception:
                continue
        for item in items:
            lp = live_by_ns.get(item["slug"])
            if lp is not None:
                item["status"] = lp.get("status", "online" if lp.get("online") else item["status"])
                item["tool_count"] = int(lp.get("tool_count", item["tool_count"]))
        return items

    items = asyncio.run(_run())
    if not items:
        typer.echo("No remote MCP servers registered.")
        return
    for u in items:
        typer.echo(
            f"{u['slug']}\tstatus={u['status']}\ttools={u['tool_count']}\t"
            f"auth={u['auth_mode']}\ttier={u['default_tier']}\turl={u['url']}"
        )


@upstream_app.command("remove")
def upstream_remove_cmd(
    slug: Annotated[str, typer.Argument(help="Remote MCP upstream slug to remove")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Remove a remote MCP server, its vault credential, and its default tier."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).remove_remote_upstream(
                slug=slug,
                principal_slug=principal,
            )

    try:
        res = asyncio.run(_run())
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Removed upstream slug={res['slug']}")


@catalog_app.command("list")
def catalog_list_cmd() -> None:
    """List available connectors from the built-in catalog (`catalog/connectors/*.toml`)."""
    entries = load_catalog()
    if not entries:
        typer.echo("No catalog connectors found.")
        return
    for c in entries:
        reg_note = " (pre-reg)" if c.auth.requires_client_registration else ""
        typer.echo(f"{c.slug}\t{c.name}\tauth={c.auth.mode}{reg_note}\turl={c.runtime.url}")


@catalog_app.command("install")
def catalog_install_cmd(
    slug: Annotated[str, typer.Argument(help="Catalog connector slug (e.g. linear, github, cfdocs)")],
    as_slug: Annotated[
        str | None,
        typer.Option("--as", help="Install under a custom prefix/slug (e.g. a second account)"),
    ] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Install a connector from the catalog and apply its manifest tier rules."""

    async def _run() -> tuple[str, dict]:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            res = await AdminService(pool).install_catalog_connector(
                catalog_slug=slug,
                as_slug=as_slug,
                catalog_dir=s.catalog_dir,
                principal_slug=principal,
            )
        return s.zos_base_url.rstrip("/"), res

    try:
        base_url, res = asyncio.run(_run())
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    manifest = res["manifest"]
    target = res["provider"]["slug"]
    typer.echo(
        f"Installed catalog connector '{manifest.slug}' as slug={target} "
        f"(auth={manifest.auth.mode}, default_tier={manifest.tiers.default})"
    )
    if manifest.auth.mode == "oauth":
        if manifest.auth.requires_client_registration:
            typer.echo(f"Next: set OAuth client credentials in {base_url}/admin/upstreams, then open {base_url}/connect/{target}")
        else:
            typer.echo(f"Next: open {base_url}/connect/{target}")
    elif manifest.auth.mode == "bearer":
        typer.echo(f"Next: run `zos upstream set-token {target}` (stdin) or paste token in {base_url}/admin/upstreams")


@module_app.command("list")
def module_list_cmd(
    catalog: Annotated[bool, typer.Option("--catalog", help="Show available catalog modules")] = False,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """List installed modules (or available catalog modules with --catalog)."""
    s = _settings()
    if catalog:
        entries = load_module_catalog(s.modules_catalog_dir)
        if not entries:
            typer.echo("No catalog modules found.")
            return
        for m in entries:
            src = m.runtime.image or m.runtime.build or "-"
            typer.echo(
                f"{m.slug}\t{m.name}\tv{m.version}\tpostgres={m.storage.postgres}\t"
                f"egress={m.network.egress}\tsource={src}"
            )
        return

    async def _run() -> list[dict]:
        async with open_pool(s.database_url) as pool:
            return await ModuleManager(pool, s).list_modules(principal_slug=principal)

    items = asyncio.run(_run())
    if not items:
        typer.echo("No modules installed.")
        return
    for item in items:
        typer.echo(
            f"{item['slug']}\tstatus={item['status']}\tv{item['version']}\t"
            f"host={item['host']}\tmcp_url={item['mcp_url'] or '-'}\tui_url={item['ui_url'] or '-'}"
        )


@module_app.command("install")
def module_install_cmd(
    slug: Annotated[str, typer.Argument(help="Catalog module slug (e.g. inbox)")],
    config: Annotated[
        list[str] | None,
        typer.Option("--config", "-c", help="Module config as KEY=VALUE (repeatable)"),
    ] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Install a module from the catalog."""
    parsed_cfg: dict[str, str] = {}
    for pair in config or []:
        if "=" not in pair:
            typer.echo(f"Error: invalid --config {pair!r}; expected KEY=VALUE", err=True)
            raise typer.Exit(code=2)
        k, v = pair.split("=", 1)
        parsed_cfg[k.strip()] = v

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await ModuleManager(pool, s).install(
                slug,
                config_values=parsed_cfg,
                principal_slug=principal,
                wait_ready=False,
            )

    try:
        res = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    mod = res["module"]
    typer.echo(
        f"Installed module slug={mod['slug']} version={mod['version']} "
        f"host={mod['host']} status={mod['status']}"
    )
    if res.get("instructions"):
        typer.echo("")
        typer.echo(res["instructions"])


@module_app.command("attach")
def module_attach_cmd(
    slug: Annotated[str, typer.Argument(help="Module slug")],
    url: Annotated[str, typer.Option("--url", help="Module MCP Streamable HTTP URL")],
    ui_url: Annotated[str | None, typer.Option("--ui-url", help="Optional module UI base URL")] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Attach a manually-run module's MCP (and optional UI) URL."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await ModuleManager(pool, s).attach(
                slug,
                mcp_url=url,
                ui_url=ui_url,
                principal_slug=principal,
            )

    try:
        mod = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"Attached module slug={mod['slug']} mcp_url={mod['mcp_url']} ui_url={mod['ui_url'] or '-'}"
    )


@module_app.command("disable")
def module_disable_cmd(
    slug: Annotated[str, typer.Argument(help="Installed module slug")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Stop a module and hide its tools while keeping its data."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await ModuleManager(pool, s).disable(slug, principal_slug=principal)

    try:
        mod = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Disabled module slug={mod['slug']}")


@module_app.command("enable")
def module_enable_cmd(
    slug: Annotated[str, typer.Argument(help="Disabled module slug")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Re-enable a disabled module."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await ModuleManager(pool, s).enable(
                slug,
                principal_slug=principal,
                wait_ready=False,
            )

    try:
        mod = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Enabled module slug={mod['slug']}")


@module_app.command("uninstall")
def module_uninstall_cmd(
    slug: Annotated[str, typer.Argument(help="Installed module slug")],
    purge: Annotated[
        bool,
        typer.Option("--purge", help="Also drop the module's Postgres schema, role, and vault secrets"),
    ] = False,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Uninstall a module (keeps its Postgres schema unless --purge is passed)."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await ModuleManager(pool, s).uninstall(
                slug,
                purge=purge,
                principal_slug=principal,
            )

    try:
        res = asyncio.run(_run())
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Uninstalled module slug={res['slug']} purge={res['purge']}")


@memory_app.command("export")
def memory_export_cmd(
    source: Annotated[str, typer.Argument(help="Importer source: claude-code | codex")],
    root: Annotated[Path | None, typer.Option("--root", help="Override source root directory")] = None,
) -> None:
    """Discover memories from a host agent directory and emit JSONL on stdout (no DB required)."""
    importer = IMPORTERS.get(source.strip().lower())
    if importer is None:
        typer.echo(f"Error: unknown source '{source}' (supported: {', '.join(sorted(IMPORTERS))})", err=True)
        raise typer.Exit(code=2)
    items = importer.discover(root)
    sys.stdout.write(dumps_jsonl(items))


@memory_app.command("import")
def memory_import_cmd(
    source: Annotated[
        str,
        typer.Argument(help="Importer source: claude-code | codex | - (read JSONL from stdin)"),
    ],
    root: Annotated[Path | None, typer.Option("--root", help="Override source root directory")] = None,
    source_label: Annotated[
        str,
        typer.Option("--source-label", help="Source label for audit events when reading from stdin ('-')"),
    ] = "stdin",
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Idempotently import agent memories into ZOS (prints counts only)."""
    clean_source = source.strip().lower()
    if clean_source == "-":
        raw = sys.stdin.read()
        try:
            items = loads_jsonl(raw)
        except Exception as exc:
            typer.echo(f"Error: invalid JSONL input: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        audit_source = source_label
    else:
        importer = IMPORTERS.get(clean_source)
        if importer is None:
            typer.echo(
                f"Error: unknown source '{source}' (supported: {', '.join(sorted(IMPORTERS))}, -)",
                err=True,
            )
            raise typer.Exit(code=2)
        items = importer.discover(root)
        audit_source = importer.source

    async def _run():
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await MemoryService(pool).import_items(
                items,
                source=audit_source,
                principal_slug=principal,
            )

    summary = asyncio.run(_run())
    typer.echo(
        f"discovered={summary.discovered} "
        f"created={summary.created} "
        f"updated={summary.updated} "
        f"unchanged={summary.unchanged}"
    )


@backup_app.command("record")
def backup_record_cmd(
    filename: Annotated[str, typer.Option("--filename", help="Encrypted backup filename")],
    size_bytes: Annotated[int, typer.Option("--size-bytes", help="Backup size in bytes")],
    completed_at: Annotated[str | None, typer.Option("--completed-at", help="ISO 8601 UTC timestamp")] = None,
    retained: Annotated[int | None, typer.Option("--retained", help="Number of retained backup files")] = None,
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Record a completed encrypted backup in the audit log for /admin Overview."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            return await AdminService(pool).record_backup(
                filename=filename,
                size_bytes=size_bytes,
                completed_at=completed_at,
                retained_count=retained,
                principal_slug=principal,
            )

    res = asyncio.run(_run())
    typer.echo(f"Recorded backup filename={res['filename']} size_bytes={res['size_bytes']}")


@app.command("hash-password")
def hash_password_cmd(
    password: Annotated[str, typer.Option("--password", prompt=True, hide_input=True)],
) -> None:
    """Generate an Argon2id hash for ZOS_OWNER_PASSWORD_HASH."""
    typer.echo(PasswordHasher().hash(password))
