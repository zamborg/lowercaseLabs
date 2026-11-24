"""ZOS command-line interface (`zos`)."""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import UUID

import typer
from argon2 import PasswordHasher

from zos.audit import record_event
from zos.config import Settings
from zos.db import open_pool, run_migrations
from zos.db.queries import (
    create_actuator_provider_and_token,
    create_bearer_client,
    find_client_by_name_or_id,
    get_principal_and_space,
    init_principal_and_space,
    mint_bearer_token_for_client,
    revoke_token,
    set_client_policy,
)

app = typer.Typer(name="zos", help="ZOS MCP router administration CLI.", no_args_is_help=True)
client_app = typer.Typer(help="Manage MCP clients.")
token_app = typer.Typer(help="Manage bearer and OAuth tokens.")
policy_app = typer.Typer(help="Manage per-client and default tool policies.")
actuator_app = typer.Typer(help="Manage dial-out actuators.")

app.add_typer(client_app, name="client")
app.add_typer(token_app, name="token")
app.add_typer(policy_app, name="policy")
app.add_typer(actuator_app, name="actuator")


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
            p, sp = await get_principal_and_space(pool, principal)
            res = await create_bearer_client(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                name=name,
                trust_ceiling=trust_ceiling,
            )
            await record_event(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                client_id=res["client"]["id"],
                event_type="client.registered",
                payload={"name": name, "kind": "bearer", "trust_ceiling": trust_ceiling},
            )
            await record_event(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                client_id=res["client"]["id"],
                event_type="token.issued",
                payload={"token_id": str(res["token_id"]), "kind": "bearer"},
            )
            return res

    res = asyncio.run(_run())
    typer.echo(f"client_id={res['client']['id']}")
    typer.echo(f"token_id={res['token_id']}")
    typer.echo(f"token={res['token']}")


@token_app.command("create")
def token_create_cmd(
    client_name: Annotated[str, typer.Argument(help="Existing bearer client name")],
    principal: Annotated[str | None, typer.Option("--principal", help="Principal slug")] = None,
) -> None:
    """Mint an additional bearer token for an existing bearer client."""

    async def _run() -> dict:
        s = _settings()
        async with open_pool(s.database_url) as pool:
            p, sp = await get_principal_and_space(pool, principal)
            res = await mint_bearer_token_for_client(
                pool,
                principal_id=p["id"],
                client_name=client_name,
            )
            await record_event(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                client_id=res["client"]["id"],
                event_type="token.issued",
                payload={"token_id": str(res["token_id"]), "kind": "bearer"},
            )
            return res

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
            p, sp = await get_principal_and_space(pool, principal)
            client_id: UUID | None = None
            token_id: UUID | None = None
            raw_token: str | None = None

            if client:
                c = await find_client_by_name_or_id(pool, principal_id=p["id"], identifier=client)
                if c is None:
                    raise ValueError(f"Client not found: {client}")
                client_id = c["id"]
            elif target:
                try:
                    token_id = UUID(target)
                except ValueError:
                    raw_token = target

            count = await revoke_token(pool, token_id=token_id, raw_token=raw_token, client_id=client_id)
            if count > 0:
                await record_event(
                    pool,
                    principal_id=p["id"],
                    space_id=sp["id"],
                    client_id=client_id,
                    event_type="token.revoked",
                    payload={"count": count, "token_id": str(token_id) if token_id else None},
                )
            return count

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
            p, sp = await get_principal_and_space(pool, principal)
            client_id: UUID | None = None
            if client:
                c = await find_client_by_name_or_id(pool, principal_id=p["id"], identifier=client)
                if c is None:
                    raise ValueError(f"Client not found: {client}")
                client_id = c["id"]

            row = await set_client_policy(
                pool,
                principal_id=p["id"],
                client_id=client_id,
                tool_pattern=pattern,
                decision=decision,
            )
            await record_event(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                client_id=client_id,
                event_type="policy.changed",
                tool=pattern,
                decision=decision,
                payload={"client": client or "*"},
            )
            return row

    row = asyncio.run(_run())
    scope_label = client or "*"
    typer.echo(f"Policy set: client={scope_label} pattern={row['tool_pattern']} decision={row['decision']}")


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
            p, sp = await get_principal_and_space(pool, principal)
            res = await create_actuator_provider_and_token(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                slug=slug,
            )
            await record_event(
                pool,
                principal_id=p["id"],
                space_id=sp["id"],
                provider_id=res["provider"]["id"],
                event_type="token.issued",
                payload={"token_id": str(res["token_id"]), "kind": "actuator", "slug": slug},
            )
            return res

    res = asyncio.run(_run())
    typer.echo(f"provider_id={res['provider']['id']}")
    typer.echo(f"token_id={res['token_id']}")
    typer.echo(f"token={res['token']}")


@app.command("hash-password")
def hash_password_cmd(
    password: Annotated[str, typer.Option("--password", prompt=True, hide_input=True)],
) -> None:
    """Generate an Argon2id hash for ZOS_OWNER_PASSWORD_HASH."""
    typer.echo(PasswordHasher().hash(password))
