"""ZOS command-line interface (`zos`).

Delegates all administrative mutations to the shared `AdminService` layer
so CLI and `/admin` share identical behavior and audit events.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from argon2 import PasswordHasher

from zos.admin_service import AdminService
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import init_principal_and_space

app = typer.Typer(name="zos", help="ZOS MCP router administration CLI.", no_args_is_help=True)
client_app = typer.Typer(help="Manage MCP clients.")
token_app = typer.Typer(help="Manage bearer and OAuth tokens.")
policy_app = typer.Typer(help="Manage per-client and default tool policies.")
actuator_app = typer.Typer(help="Manage dial-out actuators.")
provider_app = typer.Typer(help="Manage upstream providers.")

app.add_typer(client_app, name="client")
app.add_typer(token_app, name="token")
app.add_typer(policy_app, name="policy")
app.add_typer(actuator_app, name="actuator")
app.add_typer(provider_app, name="provider")


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


@app.command("hash-password")
def hash_password_cmd(
    password: Annotated[str, typer.Option("--password", prompt=True, hide_input=True)],
) -> None:
    """Generate an Argon2id hash for ZOS_OWNER_PASSWORD_HASH."""
    typer.echo(PasswordHasher().hash(password))
