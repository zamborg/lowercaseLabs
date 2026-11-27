# ZOS

**A personal, self-hosted context and MCP routing layer shared by every AI agent you use.**

> **Status: V3** (`zos-v3.2.5`). One owner per deployment, self-hosted on any small Linux server ([HOSTING.md](HOSTING.md)). It's in daily personal use and is not a hosted product.

## Why ZOS exists

Think of a good employee. They have a **brain** (intelligence) and **hands** (dexterity). They don't keep everything in their head — their notes, calendar, email, tools, and work live outside them in their workplace.

| Employee | Agent world | Owned by |
|---|---|---|
| Brain | The LLM | Model provider |
| Hands | The harness (Claude Code, Codex, ChatGPT, phone apps, …) | Client app |
| The workplace | **ZOS** | **You** |

Most AI tools fragment your workplace: Claude Code has one memory folder, Codex has another, ChatGPT and mobile apps have their own connectors, and every client needs its own tokens and permission rules.

**ZOS is the single personal endpoint behind all of them.** It never runs an agent or replaces your harness. Instead, every MCP client connects to one ZOS server that holds your shared memory, routes calls to your services and devices, enforces your approval rules, and records an audit log of everything that happens.

```
   Claude Code · Codex · ChatGPT · Phone Claude · any MCP client
                          │
                          │  OAuth 2.1 (PKCE + DCR) · Streamable-HTTP MCP
                          ▼
             ┌──────────────────────────────┐
             │     ZOS core (control plane) │  auth · policy · approvals · audit
             │        /mcp   ·   /admin     │  vault · module lifecycle
             └──┬──────────────┬─────────┬──┘
                │              │         │
                ▼              ▼         ▼
           Connectors       Modules    Devices
          (remote MCPs,   (containers  (dial in:
           catalog of 20)  ZOS runs)    Mac actuator)
                           memory · inbox · gmail · gcal
```

The core holds no domain content. Everything an agent works on reaches it as one of three kinds of upstream, and all three look the same to clients:

| Kind | Who runs it | Who does its OAuth | Examples |
|---|---|---|---|
| Connector | someone else | the ZOS core (standard MCP OAuth) | Linear, Atlassian, GitHub |
| Module | ZOS, in a locked-down container | the module itself, if needed | memory, inbox, gmail, gcal |
| Device | an MCP server on your hardware that dials in | none (device token) | `zos-mac`, any local stdio MCP server |

## What you get in V3

- **Shared memory (module, installed by default):** full-text and fuzzy memory (`memory_put/get/search/list/delete`) with client authorship. The owner view at `/admin/m/memory/` lets you browse, search, and delete.
- **Inbox (module, installed by default):** async, email-like mailboxes shared between you (`owner`), `general`, and each connected agent. The owner view at `/admin/m/inbox/` shows every mailbox.
- **Tasks (optional module):** personal todo list for the human owner (`tasks_list/get/add/update/complete`) with a mobile-friendly Today/todo-list view at `/admin/m/tasks/` (Overdue / Today / Upcoming / No date + Done).
- **Gmail & Google Calendar (optional modules):** `gmail_*` and `gcal_*`. Each module connects to Google from its own owner page with narrowly scoped, encrypted credentials.
- **Excalidraw (optional module)** gives agents and the owner shared, versioned drawings with an offline editor at `/admin/m/excalidraw/`. See [the module guide](modules/excalidraw/README.md).
- **Files (optional module):** shared POSIX-like filesystem (`files_list/tree/stat/read/write/mkdir/move/copy/delete`, real folders, atomic folder ops, and `skills/<name>/SKILL.md` sync across machines) stored in Postgres with a mobile-friendly file manager at `/admin/m/files/`. See [modules/files/README.md](modules/files/README.md).
- **Routed remote MCP servers + connector catalog:** connect any Streamable-HTTP remote MCP server (`none`, `bearer`, or `oauth` auth), or install one of 20 verified connector manifests. See [catalog/README.md](catalog/README.md).
- **Devices (optional):** turn any local stdio MCP server on your Mac or workstation into a ZOS device via `zos-device` (includes `zos-mac` for Messages, Contacts, and persistent tmux shell). See [DEVICES.md](DEVICES.md).
- **Two-layer policy:** visibility (which tools a client sees) plus a per-call allowance chain (allow, escalate to you, or deny).
- **Async approvals for phones and web clients:** Clients that support MCP elicitation prompt inline; clients that don't (like mobile Claude or ChatGPT) hold approval-gated calls briefly or return a clickable `/admin/approvals/<id>` link so you can inspect the exact arguments on your phone, tap **Approve**, and let the agent retry once.
- **Owner admin dashboard (`/admin`):** Passphrase-gated, mobile-friendly control panel for provider health, pending approvals, remote MCP servers, clients and tokens, per-tool tiers and per-client policy overrides, and the audit log.
- **Encrypted nightly backups:** Automated `pg_dump` + AES-256-CBC backups (`deploy/backup/backup.sh`) with a tested restore script and last-backup status on `/admin`.

## Quickstart

**To host ZOS on a server** (the normal path), follow [HOSTING.md](HOSTING.md). It's written so you can hand it to your coding agent: one installer, a few marked human steps, and memory + inbox installed by default.

The steps below are for running from a local checkout (development).

### 1. Start ZOS and Postgres

Requires Docker (e.g. OrbStack or Docker Desktop) and `uv` (`brew install uv`). From the `zos/` directory:

```bash
cp .env.example .env
chmod 600 .env
```

Generate secrets and add them to `.env`:

- `POSTGRES_PASSWORD`: any strong random string
- `ZOS_SECRET_KEY`: `python3 -c "import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`
- `ZOS_OWNER_PASSWORD_HASH`: run `uv run zos hash-password` and wrap the output in **single quotes** in `.env`
- `ZOS_BACKUP_KEY`: separate random passphrase for encrypted backups

Start the stack (bound strictly to `127.0.0.1:8787`) and initialize your principal and default policy tiers:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up -d --build
docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
  uv run --no-sync zos init --principal owner --display-name "Owner"
```

Verify health and open the admin dashboard:

```bash
curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool
```

Open **`http://localhost:8787/admin`** and sign in with your owner passphrase.

### 2. Expose remotely with Cloudflare Tunnel + Access (optional)

To connect ChatGPT, mobile Claude, or remote devices at `https://zos.<your-domain>`:

1. In `.env`, set `ZOS_BASE_URL=https://zos.<your-domain>`, then recreate `zos`. The redirect-host allowlist already includes `chatgpt.com`, `claude.ai`, and `agent.meta.ai`. Other agents that are rejected show up on `/admin/clients` with an **Allow** button.
2. Copy `deploy/cloudflared/config.yml.example` to `~/.cloudflared/config.yml` and fill in your `<TUNNEL-UUID>`, `<HOME>`, and `zos.<your-domain>`. Only ZOS's own paths are exposed, and `/healthz` stays local.
3. Optionally, put **Cloudflare Access** in front of the browser-only owner paths (`/admin*`, `/oauth/authorize`) as defense in depth. ZOS's own passphrase, signed cookie, CSRF, and rate limits already protect them.

### 3. Connect your clients

Point every client at `https://zos.<your-domain>/mcp` (or `http://localhost:8787/mcp` locally):

- **Claude Code:**
  ```bash
  claude mcp add --transport http zos https://zos.<your-domain>/mcp
  ```
- **Codex:**
  ```bash
  codex mcp add zos --url https://zos.<your-domain>/mcp
  ```
- **ChatGPT & mobile Claude:** add a custom MCP connector pointing at `https://zos.<your-domain>/mcp` and sign in via OAuth in your browser.
- **Headless / bearer fallback:** mint a static bearer token with:
  ```bash
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos client add my-client --bearer
  ```

### 4. Optional add-ons

- **Modules (memory, inbox, gmail, gcal):** open `/admin/modules` and click **Install**, or run `zos module install <slug>`. For Gmail and Calendar, see the Google appendix in [HOSTING.md](HOSTING.md). After installing, click **Connect** on `/admin/m/gmail/` and `/admin/m/gcal/`.
- **Remote MCP connectors:** open `/admin/upstreams` and click **Install** on any connector, or run `zos catalog list`.
- **Devices (`zos-device` + `zos-mac`):** add a device on `/admin/modules` ("Runs on your machine") or run `./deploy/launchd/install.sh` on macOS. See [DEVICES.md](DEVICES.md).

New tools appear in clients after they refresh or reconnect their ZOS connector.

## Modules

A ZOS module is a standalone MCP server (in any language) with a manifest in `catalog/modules/<slug>.toml`. ZOS owns authentication, policy, approvals, audit logging, Postgres schema provisioning, and reverse-proxying the module's human UI under `/admin/m/<slug>/`. Modules never see client OAuth/bearer tokens or core ZOS tables.

- **Isolation:** When running on Docker (`ZOS_MODULE_HOST=docker`, auto-enabled when `/var/run/docker.sock` is mounted), ZOS builds or pulls only catalog-pinned images and writes the container spec itself (`CapDrop: ALL`, `no-new-privileges`, no host mounts, memory limit, internal `zos_modules` network with egress disabled by default). Each module that requests Postgres receives its own `mod_<slug>` role and schema with zero read access to core tables (`credentials`, `events`, `tokens`). Modules may request trusted extensions (e.g. `pg_trgm`), which live in a shared `extensions` schema.
- **Signed caller identity:** Every `tools/call` forwarded to a module carries an HMAC-SHA256-signed `params._meta["zos/caller"]` envelope (`principal_id`, `client_id`, `client_name`, `tool`, `args_digest`, `ts`) signed with the module's per-module `ZOS_MODULE_SECRET`. Unsigned calls (like `tools/list`) are system calls. Proxied UI requests under `/admin/m/<slug>/` strip the owner admin cookie and send signed `X-ZOS-Caller` + `X-ZOS-Caller-Signature` headers (`client_name = "owner"`) plus `X-Forwarded-Prefix`.
- **Manual host fallback:** Without a Docker socket (or with `ZOS_MODULE_HOST=manual`), `zos module install <slug>` prints container run instructions and a coding-agent prompt; finish with `zos module attach <slug> --url <mcp-url> [--ui-url <ui-url>]`.
- **How to write a module:** Create a standalone server that exposes Streamable-HTTP MCP on `MCP_PORT` (and optionally a mobile-friendly HTML UI on `UI_PORT`), copy the stdlib-only verifier [zos_module_sdk/caller.py](zos_module_sdk/caller.py) to verify `_meta["zos/caller"]` and `X-ZOS-Caller`, read `ZOS_MODULE_DATABASE_URL` and `ZOS_MODULE_SECRET` from env, and add a manifest under `catalog/modules/<slug>.toml`. **Owner visibility rule:** every module that stores data must provide an owner view under `/admin/m/<slug>` covering **all** of its underlying state (e.g. Inbox shows every mailbox and client-to-client thread, and owner reads never mutate clients' read state). See `modules/inbox/` for a complete reference implementation and [catalog/README.md](catalog/README.md) for the manifest schema.

## Security model in brief

- **Tool facts vs. client decisions:** Every tool has a global tier (`read`, `write`, or `external`; unknown tools default to `external`). By default `read` and `write` are allowed and `external` requires approval, with per-client overrides on `/admin/policy`. Clients marked `untrusted` can never execute approval-gated tools.
- **Server-enforced confirmation:** Approval happens inside ZOS — via MCP elicitation when supported, or via CSRF-protected `/admin/approvals/<id>` otherwise. Pending tool arguments are encrypted with AES-256-GCM at rest and purged immediately once decided or expired.
- **Encrypted credential vault:** Upstream OAuth tokens and bearer secrets are encrypted with AES-256-GCM using `ZOS_SECRET_KEY` (never stored in plaintext and never exposed to agents).
- **Append-only audit log:** Every tool call, auth failure, policy change, and approval decision is logged to `events` using SHA-256 argument digests — raw tool arguments are never stored in the audit log.
- **Defense in depth:** Global passphrase rate limiting across all login/connect forms, DCR redirect host allowlist and client caps, `127.0.0.1`-only container port bindings, and a dedicated Python binary for Mac actuator Full Disk Access.

## Further reading

- [HOSTING.md](HOSTING.md): host your own ZOS (agent-executable)
- [V2.md](V2.md): architecture direction and the V3 final shape (core, connectors, modules, devices)
- [PLAN.md](PLAN.md): original V1 architecture and component design
- [INVARIANTS.md](INVARIANTS.md): architectural and security invariants, mapped to the tests that enforce them
- [MODULES.md](MODULES.md): Module Contract v1.1
- [TESTING.md](TESTING.md) — Full operations, host migration, backup/restore, and testing runbook
- [SCHEMA.md](SCHEMA.md) — Postgres schema reference
- [DECISIONS.md](DECISIONS.md) — Append-only architectural decision log
- [actuator/README.md](actuator/README.md) — Mac actuator daemon setup and tools
- [catalog/README.md](catalog/README.md) — Remote MCP connector catalog and manifest format
