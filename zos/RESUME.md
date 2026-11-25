# ZOS — Current Status & Handoff

Read [README.md](README.md) for the user-facing overview and [PLAN.md](PLAN.md) for architecture details.

## What exists now (V1 shipped)

ZOS V1 is complete, running self-hosted on Docker Compose + Postgres 17 behind a Cloudflare Named Tunnel, and verified end-to-end across desktop CLI clients, ChatGPT, and mobile Claude:

- **Core MCP server & OAuth 2.1 AS:** Streamable-HTTP MCP at `/mcp`, PKCE + DCR (with redirect allowlist, hourly rate limit, and client cap), bearer token fallback, client soft-disable (`zos client disable`), and spec-native `InitializeResult.instructions`.
- **Hosted memory:** `memory_put`, `memory_get`, `memory_search`, `memory_list`, `memory_delete` with author attribution and host memory import (`deploy/scripts/import-memories.sh`).
- **Google upstream:** Shared encrypted Google credential backing `gcal` (hot-switchable `api | mcp`) and `gmail` (REST search/read/MIME-threaded send).
- **Mac actuator (`actuator/`):** Standalone daemon running on a dedicated Python 3.12 copy (`~/.zos-actuator/python/bin/python3.12`) exposing `mac_messages_search` (with AddressBook contact resolution) and `mac_shell_run` (isolated `tmux -L zos`).
- **Routed remote MCPs & connector catalog (`catalog/`):** Register any Streamable-HTTP remote MCP (`none`, `bearer`, `oauth`) or install from 20 empirically verified manifests in `catalog/connectors/*.toml` (nothing installed by default).
- **Policy & async approvals:** Tool tiers (`read | write | external`), per-client overrides, MCP elicitation when supported, and AEAD-encrypted async approvals (`/admin/approvals/<id>`) for non-elicitation clients (phones, ChatGPT).
- **Owner admin dashboard (`/admin`):** Overview, Approvals, MCP Servers, Clients, Tools & Policy, and Activity audit log.
- **Encrypted nightly backups:** `deploy/backup/backup.sh` and `deploy/backup/restore.sh` with LaunchAgent template and Overview status card.

## How to run, onboard, and operate

1. **Deploy / start locally:** copy `.env.example` → `.env`, fill secrets, run `docker compose --env-file .env -f deploy/docker-compose.yml up -d --build`, then `zos init`. See [README.md](README.md) for the quickstart and [TESTING.md](TESTING.md) for the full operations runbook.
2. **Mac actuator:** run `./deploy/launchd/install.sh`, grant Full Disk Access to `~/.zos-actuator/python/bin/python3.12`, set the token from `zos actuator add mac`, and load the LaunchAgents. See [actuator/README.md](actuator/README.md).
3. **Public exposure:** configure Cloudflare Named Tunnel (`deploy/cloudflared/config.yml.example`) and Cloudflare Access on `/admin*`, `/oauth/authorize`, and `/connect/*`.
4. **Tests & lint:**
   ```bash
   uv run ruff check && uv run pytest
   (cd actuator && uv run ruff check && uv run pytest)
   ```

## Next direction

[V2.md](V2.md) outlines the unscheduled V2 direction (repackaging services as installable modules, chat, whiteboard, app hosting, skills, and files). Do not expand scope without an explicit decision recorded in [DECISIONS.md](DECISIONS.md).
