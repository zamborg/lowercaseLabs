# ZOS — Plan (V1)

Status: **V1 shipped** (2026-11-25). Architecture direction for V2+ (installable modules, files, chat, skills): [V2.md](V2.md). Schema: [SCHEMA.md](SCHEMA.md). Operations & testing runbook: [TESTING.md](TESTING.md). Decision log: [DECISIONS.md](DECISIONS.md). Earlier runtime-shaped notes are archived in `later/`.

## What ZOS is

A personal, self-hostable **context and routing engine for MCP calls**. Any MCP client (Claude Code, Codex, ChatGPT, phone apps, local LLM UIs) connects to one ZOS MCP endpoint. Behind it, ZOS hosts shared memory and routes tool calls to upstream services (Google, any remote MCP server) and local actuators (such as a Mac daemon), enforcing permissions, approvals, and an append-only audit trail on every call.

Agents run outside ZOS. ZOS never runs an agent.

**The boundary.** ZOS is everything behind the MCP interface and nothing in front of it. Which harness, model, or UI the agent runs in, and how it is configured (`CLAUDE.md`, `AGENTS.md`, skills, prompts), is the agent's business. ZOS owes every MCP client one thing: a correct, spec-compliant MCP server. We design and test against the MCP spec, never against one client's quirks.

Marquee problem: memories, credentials, and MCP servers diverge across Claude Code, Codex, ChatGPT, and phone apps. With ZOS, every client points at one endpoint and shares the same layer.

## Schematic

```
   any MCP client (Claude Code · Codex · ChatGPT · Phone · …)
                  │  OAuth 2.1 (PKCE + DCR) · Streamable-HTTP MCP
       ┌──────────┴───────────┐
       │      ZOS router      │  auth · policy · approvals · namespacing · audit
       │   (/mcp  ·  /admin)  │
       └──┬────────┬────────┬─┴──────────────┐
    hosted│        │        │                │
    memory  built-in Google │           routed remote MCPs
            (gcal MCP/REST, │           (20-connector catalog
             gmail REST)    │            + custom none/bearer/oauth)
                            │
                     Mac actuator (dials out over WS, speaks MCP;
                     Messages search + contact names, tmux shell)
                            │
             Postgres 17 (self-hosted via Docker Compose)
```

## Components (V1)

- **MCP edge + auth.** Streamable-HTTP MCP at `/mcp` with spec-native `InitializeResult.instructions`. Clients sign in via MCP OAuth 2.1 (PKCE + dynamic client registration with redirect host allowlist, hourly rate limit, and client cap) against ZOS's single-user authorization server. Static bearer tokens (`zos client add <name> --bearer`) serve headless clients and actuators. Global passphrase rate limiting protects `/admin/login`, `/oauth/authorize`, `/connect/google`, and `/connect/<slug>`.
- **Owner admin dashboard (`/admin`).** Server-rendered, passphrase-gated, CSRF-protected control panel with mobile-friendly pages:
  - **Overview:** live provider status, Google connection, `gcal` `api|mcp` hot backend toggle, pending approvals badge, last encrypted backup status.
  - **Approvals (`/admin/approvals`, `/admin/approvals/<id>`):** review decrypted arguments and approve or deny calls from clients without MCP elicitation.
  - **MCP Servers (`/admin/upstreams`):** install from the 20-connector catalog, register custom remote MCPs (`none`, `bearer`, or `oauth`), manage credentials, disconnect, or remove live without restart.
  - **Clients (`/admin/clients`):** mint bearer tokens, revoke tokens, disable retired clients (preserving audit and memory history), and prune OAuth clients.
  - **Tools & Policy (`/admin/policy`):** per-tool tier editor (`read | write | external`) and per-client decision matrix (`allow | require_approval | deny`).
  - **Activity (`/admin/activity`):** cursor-paginated append-only audit trail (SHA-256 argument digests only; raw arguments are never stored in `events`).
  Mutations share one service layer (`AdminService`) with the `zos` CLI and are audited.
- **Router.** Merges tools from all online providers, namespaces them, filters by policy, routes calls, and writes one audit event per call. Always-on `zos_status` reports live provider health.
- **Providers — the one seam behind the router:**
  `ToolProvider { namespace, is_online(), list_tools(), call_tool(name, args, ctx) }`
  — `namespace` = provider slug = tool prefix (`memory_*`, `gcal_*`, `gmail_*`, `mac_*`, `<slug>_*`).
  - *hosted* — in-process, Postgres-backed (`memory`).
  - *upstream* — both first-party adapters (`gmail` REST, `gcal` MCP proxy with REST fallback, sharing one encrypted Google credential) and generic routed remote MCP servers (`McpUpstreamProvider` over Streamable HTTP with `none`, `bearer`, or `oauth` auth via `OAuthClientProvider`, plus checked-in `catalog/connectors/*.toml` manifests). Tokens live in the AEAD vault and are never exposed to agents.
  - *actuator* — user-run daemon dials out over WebSocket to `/actuator` and speaks MCP as a server; ZOS is the MCP client on that socket.
- **Visibility.** Actuator and remote MCP tools are listed **only while connected**; ZOS emits `notifications/tools/list_changed` on connect, disconnect, hot add/remove, or upstream tool list changes. Stale actuator calls return `ACTUATOR_OFFLINE`; a socket lost mid-call returns `ACTUATOR_DISCONNECTED_MID_CALL` (do not retry blindly).
- **Policy — tool facts vs client decisions, kept separate.**
  - *Tier* is a global fact about a tool: `read | write | external` (unknown → `external`). It drives MCP tool annotations and default decisions. Never downgraded merely to grant one client access.
  - *Decision* per client: `allow | deny | require_approval`. Default from tier (`read` → `allow`, `write` → `allow`, `external` → `require_approval`); per-client and all-client overrides by tool pattern. `trust_ceiling = 'untrusted'` downgrades `require_approval` → `deny`.
- **Confirmation (`require_approval`).** Enforced server-side by ZOS using two strategies behind the same decision:
  - When a client declares MCP elicitation capability: asks via `elicitation/create` showing the exact untruncated tool call.
  - When a client lacks elicitation (e.g., phone Claude or ChatGPT): creates a `pending_approvals` row with AEAD-encrypted arguments, holds the call up to `ZOS_APPROVAL_WAIT_S` (default 45s) for an in-window approve/deny on `/admin/approvals/<id>`, or returns the approval link so the owner can approve and the assistant can retry once within the 10-minute TTL. Encrypted arguments are purged immediately on decision, consumption, or expiration.
- **Hosted memory.** `memory_put`, `memory_get`, `memory_search`, `memory_list`, `memory_delete` (Postgres full-text + trigram, with `author` attribution and timestamps), plus host memory import (`zos memory export` / `zos memory import`).
- **Audit & Backups.** Append-only `events` table (digests only, no raw arguments). Encrypted nightly `pg_dump` backups via `deploy/backup/backup.sh` and verified restore via `deploy/backup/restore.sh`.

## V1 built-in providers & catalog

- **hosted `memory`:** shared personal memory store (`memory_put/get/search/list/delete`).
- **upstream `gcal` + `gmail`:** share one Google OAuth credential (`/connect/google`).
  - `gcal` supports hot-switching between Google's Calendar MCP (`mcp`) and our Calendar REST adapter (`api`) with zero restart.
  - `gmail` uses our Gmail REST adapter (bounded metadata search, read, MIME-threaded send).
- **actuator `mac`:** standalone macOS daemon installed with a dedicated Python 3.12 copy (`~/.zos-actuator/python/bin/python3.12`) exposing:
  - `mac_messages_search` — read-only FTS5 search over `~/Library/Messages/chat.db` with AddressBook contact name resolution.
  - `mac_shell_run` — run commands in an isolated persistent local tmux session (`tmux -L zos`).
- **routed remote MCPs + connector catalog:** add any Streamable HTTP remote MCP server (`none`, `bearer`, `oauth`) or install from 20 empirically verified manifests in `catalog/connectors/*.toml` (nothing installed by default; multi-instance via `--as`).

## Tool naming

`<prefix>_<tool>`, charset `[a-zA-Z0-9_-]`, prefixes match `^[a-z][a-z0-9]{1,15}$` (no `_`), total ≤ 48 chars (deterministic truncate + 6-char hash on overflow). Upstream tool names are sanitized to `[a-zA-Z0-9_-]+` while forwarding raw names upstream, and exposed-name collisions are rejected. Prefix `zos` is reserved for router-owned tools (`zos_status`).

## Hosting & stack

- Python 3.12, `uv`, Starlette, official `mcp` Python SDK (`mcp.server.lowlevel.Server`), `psycopg` 3, plain numbered SQL migrations. One container + Postgres 17.
- **Self-host today:** Docker Compose (`deploy/docker-compose.yml`, bound strictly to `127.0.0.1`) on a laptop or home server + Cloudflare Named Tunnel + Cloudflare Access on browser-only owner paths (`/admin*`, `/oauth/authorize`, `/connect/*`). `/actuator` and `/healthz` stay local-only.
- **Vault:** AEAD (AES-256-GCM) with key from `ZOS_SECRET_KEY`, never stored in Postgres; `key_version` for rotation; per-credential async refresh lock.
- **Spaces:** V1 uses one default space per principal; every client binds to it.

## Direction (V2+)

See [V2.md](V2.md) for the next phase: turning built-in providers and new services (chat, whiteboard, app hosting, skills, files) into installable modules behind the same MCP core plane, plus eventual push notifications and vector search.
