# ZOS

**A personal, self-hosted context and MCP routing layer shared by every AI agent you use.**

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
             ┌────────────────────────┐
             │       ZOS Router       │  auth · policy · approvals · audit
             │    /mcp   ·   /admin   │
             └──┬────────┬────────┬───┴──────────────┐
                │        │        │                  │
                ▼        ▼        ▼                  ▼
             Hosted    Google    Mac Actuator     Remote MCPs
             Memory   Calendar   (Messages +      (20-connector
            (Postgres) & Gmail   tmux shell)       catalog + custom)
                │
                ▼
           Postgres 17 (encrypted vault · append-only audit log)
```

## What you get in V1

- **Shared memory:** Postgres-backed full-text and fuzzy memory (`memory_put`, `memory_get`, `memory_search`, `memory_list`, `memory_delete`) with client authorship, plus a host importer for existing Claude Code and Codex memory files.
- **Google Calendar & Gmail:** One encrypted Google sign-in powers both `gcal_*` (hot-switchable between Google's Calendar MCP and a REST adapter) and `gmail_*` (search, read, and MIME-threaded send).
- **Mac actuator:** An optional local macOS daemon that dials out to ZOS over WebSocket to expose read-only iMessage search (with AddressBook contact names) and a persistent isolated `tmux` shell. See [actuator/README.md](actuator/README.md).
- **Routed remote MCP servers + connector catalog:** Connect any Streamable-HTTP remote MCP server (`none`, `bearer`, or `oauth` auth), or install from 20 verified connector manifests (GitHub, Linear, Notion, Cloudflare, Sentry, Stripe, Vercel, Figma, Supabase, Atlassian, Slack, Asana, and more). Nothing from the catalog is installed by default. See [catalog/README.md](catalog/README.md).
- **Async approvals for phones and web clients:** Clients that support MCP elicitation prompt inline; clients that don't (like mobile Claude or ChatGPT) hold approval-gated calls briefly or return a clickable `/admin/approvals/<id>` link so you can inspect the exact arguments on your phone, tap **Approve**, and let the agent retry once.
- **Owner admin dashboard (`/admin`):** Passphrase-gated, mobile-friendly control panel for provider health, pending approvals, remote MCP servers, clients and tokens, per-tool tiers and per-client policy overrides, and the audit log.
- **Encrypted nightly backups:** Automated `pg_dump` + AES-256-CBC backups (`deploy/backup/backup.sh`) with a tested restore script and last-backup status on `/admin`.

## Quickstart

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

1. In `.env`, set `ZOS_BASE_URL=https://zos.<your-domain>` and (for ChatGPT) `ZOS_ALLOWED_REDIRECT_HOSTS=chatgpt.com`, then recreate `zos`.
2. Copy `deploy/cloudflared/config.yml.example` to `~/.cloudflared/config.yml` and fill in your `<TUNNEL-UUID>`, `<HOME>`, and `zos.<your-domain>`. Only `/mcp`, `/.well-known/*`, `/oauth/*`, `/admin*`, and `/connect/*` are exposed; `/actuator` and `/healthz` stay strictly local.
3. Put **Cloudflare Access** (allowing only your email) in front of the browser-only owner paths (`/admin*`, `/oauth/authorize`, `/connect/*`) while leaving `/mcp`, `/.well-known/*`, `/oauth/register`, and `/oauth/token` accessible to MCP clients.

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

- **Google Calendar & Gmail:** add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to `.env`, recreate `zos`, and open `/connect/google`.
- **Mac actuator:** run `./deploy/launchd/install.sh` — see [actuator/README.md](actuator/README.md).
- **Remote MCP connectors:** open `/admin/upstreams` and click **Install** on any connector, or run `zos catalog list`.

## Security model in brief

- **Tool facts vs. client decisions:** Every tool has a global tier (`read`, `write`, or `external`; unknown tools default to `external`). By default `read` and `write` are allowed and `external` requires approval, with per-client overrides on `/admin/policy`. Clients marked `untrusted` can never execute approval-gated tools.
- **Server-enforced confirmation:** Approval happens inside ZOS — via MCP elicitation when supported, or via CSRF-protected `/admin/approvals/<id>` otherwise. Pending tool arguments are encrypted with AES-256-GCM at rest and purged immediately once decided or expired.
- **Encrypted credential vault:** Upstream OAuth tokens and bearer secrets are encrypted with AES-256-GCM using `ZOS_SECRET_KEY` (never stored in plaintext and never exposed to agents).
- **Append-only audit log:** Every tool call, auth failure, policy change, and approval decision is logged to `events` using SHA-256 argument digests — raw tool arguments are never stored in the audit log.
- **Defense in depth:** Global passphrase rate limiting across all login/connect forms, DCR redirect host allowlist and client caps, `127.0.0.1`-only container port bindings, and a dedicated Python binary for Mac actuator Full Disk Access.

## Further reading

- [PLAN.md](PLAN.md) — V1 architecture and component design
- [V2.md](V2.md) — Future direction (installable modules, chat, whiteboard, app hosting, skills)
- [TESTING.md](TESTING.md) — Full operations, onboarding, backup/restore, and testing runbook
- [SCHEMA.md](SCHEMA.md) — Postgres schema reference
- [DECISIONS.md](DECISIONS.md) — Append-only architectural decision log
- [actuator/README.md](actuator/README.md) — Mac actuator daemon setup and tools
- [catalog/README.md](catalog/README.md) — Remote MCP connector catalog and manifest format
