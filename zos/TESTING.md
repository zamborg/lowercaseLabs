# ZOS V1 — Operations & Testing Runbook

Complete runbook for deploying, configuring, operating, backing up, and testing ZOS V1.
For a conceptual overview and quickstart, start with [README.md](README.md).

---

## 1. Initial Setup (Docker Compose + Postgres 17)

From the `zos/` directory:

1. **Create `.env`** from `.env.example` (keep permissions `0600`; `.env` is gitignored):
   ```bash
   cp .env.example .env
   chmod 600 .env
   ```
   Fill in:
   - `POSTGRES_PASSWORD`: strong random password
   - `ZOS_SECRET_KEY`: 32-byte urlsafe base64 key:
     ```bash
     python3 -c "import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
     ```
   - `ZOS_OWNER_PASSWORD_HASH`: Argon2id hash generated via `uv run zos hash-password` — **wrap in single quotes** in `.env` so Docker Compose does not interpolate `$` characters.
   - `ZOS_BACKUP_KEY`: separate random passphrase for nightly encrypted `pg_dump` backups.
   - `ZOS_BASE_URL`: `http://localhost:8787` for local-only, or `https://zos.<your-domain>` when behind a Cloudflare Named Tunnel.

2. **Start the stack** (bound strictly to `127.0.0.1:8787`):
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml up -d --build
   ```

3. **Initialize principal, space, memory provider, and default tool tiers:**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
     uv run --no-sync zos init --principal owner --display-name "Owner"
   ```

4. **Verify health and admin login:**
   ```bash
   curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool
   ```
   Open `http://localhost:8787/admin` (or `https://zos.<your-domain>/admin`) and sign in with your owner passphrase.

---

## 2. Google Calendar & Gmail (Modules)

`gmail` and `gcal` run as standalone container modules (`catalog/modules/gmail.toml`, `catalog/modules/gcal.toml`, built from the shared `modules/google` image) with `[network] egress = true`, isolated Postgres schemas (`mod_gmail`, `mod_gcal`), and narrower per-service Google OAuth scopes.

See [HOSTING.md — Appendix: Optional Google Modules](HOSTING.md#appendix--optional-google-modules-gmail--gcal) for the full Google Cloud Console setup, redirect URIs (`/admin/m/gmail/callback` and `/admin/m/gcal/callback`), install commands, and live `api|mcp` backend switching on `/admin/m/gcal/`.

---

## 3. Public Exposure (Cloudflare Named Tunnel + Access)

To reach ZOS from ChatGPT, phone Claude, or other remote clients at `https://zos.<your-domain>`:

1. **Update `.env` and recreate:**
   - `ZOS_BASE_URL=https://zos.<your-domain>` (enables `Secure` cookies, HTTPS redirect derivation, and RFC 8707 OAuth resource binding).
   - `ZOS_ALLOWED_REDIRECT_HOSTS=chatgpt.com` (default is empty = loopback only; comma-separated external HTTPS hosts allowed at `/oauth/register`).
   - Recreate `zos`:
     ```bash
     docker compose --env-file .env -f deploy/docker-compose.yml up -d --force-recreate zos
     ```
2. **Add the HTTPS Google callback** (`https://zos.<your-domain>/connect/google/callback`) in Google Cloud Console if using Google.
3. **Configure `cloudflared` ingress allowlist:**
   - Copy `deploy/cloudflared/config.yml.example` to `~/.cloudflared/config.yml` and substitute `<TUNNEL-UUID>`, `<HOME>`, and `zos.<your-domain>`.
   - Only `/mcp`, `/.well-known/*`, `/oauth/*`, `/admin*`, and `/connect/*` are routed to `http://localhost:8787`. `/actuator` and `/healthz` return `404` on the public internet.
   - Install the LaunchAgent from `deploy/launchd/com.zos.cloudflared.plist` (substituting `{{HOME}}` and `{{TUNNEL_NAME}}`).
4. **Add Cloudflare Access (Zero Trust) on browser-only owner paths:**
   - Protect with an email allow policy:
     - `zos.<your-domain>/admin*`
     - `zos.<your-domain>/oauth/authorize`
     - `zos.<your-domain>/connect/*`
   - Leave machine-to-machine OAuth & MCP paths **bypassing Access** so clients can discover, register, exchange tokens, and call tools:
     - `zos.<your-domain>/mcp`
     - `zos.<your-domain>/.well-known/*`
     - `zos.<your-domain>/oauth/register`
     - `zos.<your-domain>/oauth/token`
5. **Active abuse protections:**
   - Shared global passphrase rate limiter (5 failures/min, 20/hr, exponential backoff, audited as `auth.failed`) across `/admin/login`, `/oauth/authorize`, `/connect/google`, and `/connect/<slug>`.
   - `/oauth/register` rate-limited (default 10/hr) and capped at 50 active OAuth clients (`ZOS_DCR_MAX_PER_HOUR`, `ZOS_MAX_OAUTH_CLIENTS`).

---

## 4. Connecting & Managing MCP Clients

### OAuth 2.1 (recommended)

Point any MCP client at `https://zos.<your-domain>/mcp` (or `http://localhost:8787/mcp` locally):

- **Claude Code:**
  ```bash
  claude mcp add --transport http zos https://zos.<your-domain>/mcp
  ```
- **Codex:**
  ```bash
  codex mcp add zos --url https://zos.<your-domain>/mcp
  ```
- **ChatGPT / Claude mobile:** add a custom MCP connector pointing at `https://zos.<your-domain>/mcp` and complete OAuth sign-in in the browser.

### Bearer token fallback (headless clients)

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
  uv run --no-sync zos client add my-script --bearer
```

Pass `Authorization: Bearer <token>` to `/mcp`.

### Retiring or pruning clients

- **Disable any client (recommended; preserves audit attribution and memory authorship):**
  ```bash
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos client disable <client-name-or-id>
  ```
  (Or click **Disable** on `/admin/clients`.) Disabling revokes all active tokens, blocks future authentication, hides the client from `/admin/clients` and `/admin/policy`, and keeps historical `events` and `memories` authorship intact.
- **Prune an unused OAuth DCR client:** click **Prune** on `/admin/clients` or run `zos client delete <oauth-client>`.

---

## 5. Mac Actuator (Messages + Local Tmux Shell)

See [actuator/README.md](actuator/README.md) for full details. Quick steps:

1. **Create an actuator token on ZOS:**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
     uv run --no-sync zos actuator add mac
   ```
2. **Install with a dedicated Python 3.12 interpreter:**
   ```bash
   ./deploy/launchd/install.sh
   ```
   This copies uv's CPython 3.12 directory into `~/.zos-actuator/python/`, builds `~/.zos-actuator/venv` against `~/.zos-actuator/python/bin/python3.12`, creates `~/.zos-actuator/config.toml` (`chmod 600`), and installs LaunchAgents (`com.zos.tmux` and `com.zos.actuator`).
3. **Edit `~/.zos-actuator/config.toml`** and paste the token (keep `url = "ws://127.0.0.1:8787/actuator"` — `/actuator` is intentionally local-only).
4. **Grant Full Disk Access** (for the `mac_messages_*` / `mac_contacts_*` read tools); `mac_messages_send` additionally needs **Automation → Messages**, which macOS prompts for on first send:
   - **System Settings → Privacy & Security → Full Disk Access → `+`** (`Cmd+Shift+G`) → add:
     ```text
     ~/.zos-actuator/python/bin/python3.12
     ```
   - Isolating the binary to `~/.zos-actuator/python/` ensures Full Disk Access is NOT granted to uv's shared Python interpreter.
5. **Start the LaunchAgents:**
   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.tmux.plist || true
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist || true
   ```

---

## 6. Remote MCP Servers & Connector Catalog

Manage connectors from **`/admin/upstreams`** ("MCP Servers") or via the CLI. Nothing from the catalog is installed by default.

- **List & install from catalog:**
  ```bash
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos catalog list
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos catalog install linear
  # Multi-instance install under a custom prefix:
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos catalog install linear --as worklinear
  ```
- **Add or manage a custom remote MCP server:**
  ```bash
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos upstream add github --url https://api.githubcopilot.com/mcp/ --auth bearer --default-tier external
  # Pass bearer tokens via stdin (never argv):
  printf '%s\n' "$GITHUB_PAT" | docker compose --env-file .env -f deploy/docker-compose.yml exec -T zos \
    uv run --no-sync zos upstream set-token github
  # For OAuth upstreams, open the printed /connect/<slug> URL on desktop or phone:
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
    uv run --no-sync zos upstream connect linear
  ```
- **Per-tool tiers & policies:** override any tool's tier (`read | write | external`) or per-client decision (`allow | require_approval | deny`) on **`/admin/policy`**. Owner-edited tiers are never overwritten when an upstream updates its tool list.

### Catalog Empirical Verification Log (2026-11-24)

All 20 entries in `catalog/connectors/*.toml` were probed with unauthenticated HTTP requests only (`POST` initialize + PRM + AS metadata):

| Slug | Endpoint URL | Verified Date | HTTP Status | Auth Mode | DCR (`registration_endpoint`) | Notes |
|---|---|---|---|---|---|---|
| `github` | `https://api.githubcopilot.com/mcp/` | 2026-11-24 | 401 + PRM | `bearer` | No (`github.com/login/oauth`) | Uses fine-grained PAT (`token_help` included) |
| `linear` | `https://mcp.linear.app/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`mcp.linear.app/register`) | Ready for 1-click OAuth connect |
| `notion` | `https://mcp.notion.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`mcp.notion.com/register`) | Ready for 1-click OAuth connect |
| `cloudflare` | `https://mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`mcp.cloudflare.com/register`) | Unified Cloudflare API MCP |
| `cfdocs` | `https://docs.mcp.cloudflare.com/mcp` | 2026-11-24 | 200 OK | `none` | N/A | Public read-only docs |
| `cfbindings` | `https://bindings.mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`bindings.mcp.cloudflare.com/register`) | Workers Bindings |
| `cfbuilds` | `https://builds.mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`builds.mcp.cloudflare.com/register`) | Workers Builds |
| `cfobs` | `https://observability.mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`observability.mcp.cloudflare.com/register`) | Workers Observability |
| `cfradar` | `https://radar.mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`radar.mcp.cloudflare.com/register`) | Cloudflare Radar (`default=read`) |
| `cfdns` | `https://dns-analytics.mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`dns-analytics.mcp.cloudflare.com/register`) | DNS Analytics (`default=read`) |
| `cfaudit` | `https://auditlogs.mcp.cloudflare.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`auditlogs.mcp.cloudflare.com/register`) | Audit Logs (`default=read`) |
| `atlassian` | `https://mcp.atlassian.com/v2/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`auth.atlassian.com/.../dcr/register`) | Jira & Confluence Rovo MCP v2 |
| `sentry` | `https://mcp.sentry.dev/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`mcp.sentry.dev/oauth/register`) | Ready for 1-click OAuth connect |
| `stripe` | `https://mcp.stripe.com/` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`access.stripe.com/mcp/oauth2/register`) | Ready for 1-click OAuth connect |
| `vercel` | `https://mcp.vercel.com/` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`api.vercel.com/login/oauth/register`) | Ready for 1-click OAuth connect |
| `figma` | `https://mcp.figma.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`api.figma.com/v1/oauth/mcp/register`) | Ready for 1-click OAuth connect |
| `supabase` | `https://mcp.supabase.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` | Yes (`api.supabase.com/platform/oauth/apps/register`) | Ready for 1-click OAuth connect |
| `context7` | `https://mcp.context7.com/mcp` | 2026-11-24 | 200 OK | `none` | N/A | Public read-only library docs |
| `slack` | `https://mcp.slack.com/mcp` | 2026-11-24 | 401 + PRM | `oauth` (pre-reg) | No (`mcp.slack.com`) | `requires_client_registration = true` |
| `asana` | `https://mcp.asana.com/v2/mcp` | 2026-11-24 | 401 + PRM | `oauth` (pre-reg) | No (`app.asana.com`) | `requires_client_registration = true` (legacy `/sse` is 410) |

See [catalog/README.md](catalog/README.md) for how to add new connectors.

---

## 7. Async Approvals (Phone & Non-Elicitation Clients)

- Clients declaring MCP elicitation capability use interactive `elicitation/create` prompts unchanged.
- Clients without elicitation (such as mobile Claude or ChatGPT) automatically use the async approval flow on any `require_approval` call:
  1. ZOS encrypts the tool arguments in `pending_approvals` using the AEAD vault and holds the call up to `ZOS_APPROVAL_WAIT_S` (default `45` seconds).
  2. If you approve on **`/admin/approvals/<id>`** while held, the tool executes immediately and returns the real result.
  3. If the hold window times out, ZOS returns a tool error containing `<ZOS_BASE_URL>/admin/approvals/<id>`. Open the link (on phone or desktop), review the exact escaped arguments, tap **Approve**, and tell the assistant to retry the same call.
  4. Retrying the exact `(client, tool, args_digest)` within the 10-minute TTL atomically consumes the approval once and executes. A second retry is re-gated.
  5. Encrypted arguments are purged immediately once decided, consumed, or expired, and never appear in `events`.

---

## 8. Importing Existing Host Memories

Import existing Claude Code and Codex local memory files into ZOS (idempotent; prints counts only):

```bash
./deploy/scripts/import-memories.sh
```

Attributed to the dedicated `memory-import` client and audited as `memory.imported`.

---

## 9. Encrypted Backups & Restore

- **Run a backup manually:**
  ```bash
  ./deploy/backup/backup.sh
  ```
  Dumps Postgres (`pg_dump -Fc`), encrypts via `/usr/bin/openssl enc -aes-256-cbc -pbkdf2 -salt` using `ZOS_BACKUP_KEY`, saves to `~/Library/Mobile Documents/com~apple~CloudDocs/zos-backups/` (override with `ZOS_BACKUP_DIR`), keeps the newest 14 files, and records metadata (`backup.completed`) shown on `/admin` Overview.
- **Schedule nightly backups (macOS):** render `deploy/launchd/com.zos.backup.plist` (substituting `{{HOME}}` and `{{ZOS_DIR}}`) into `~/Library/LaunchAgents/com.zos.backup.plist` and load with `launchctl bootstrap`.
- **Restore into a test or live Postgres container:**
  ```bash
  # Test restore into an isolated container:
  ./deploy/backup/restore.sh ~/Library/Mobile\ Documents/com~apple~CloudDocs/zos-backups/zos-backup-<ts>.dump.enc \
    --container zos-pg-test --db zos_restore_test --user zos

  # Restore into live compose Postgres (stop zos first):
  docker compose --env-file .env -f deploy/docker-compose.yml stop zos
  ./deploy/backup/restore.sh ~/Library/Mobile\ Documents/com~apple~CloudDocs/zos-backups/zos-backup-<ts>.dump.enc
  docker compose --env-file .env -f deploy/docker-compose.yml start zos
  ```

---

## 10. Modules & Inbox

`deploy/docker-compose.yml` mounts `/var/run/docker.sock` into `zos` and declares the internal `zos_modules` network (shared by `zos`, `postgres`, and installed modules) plus `zos_modules_egress` (attached only when a module declares `network.egress = true`).

- **Install from UI:** Open `/admin/modules` and click **Install** next to **Inbox**. Once running, **Inbox** appears in the `/admin` nav bar and Overview page at `/admin/m/inbox/`.
- **Install / manage from CLI:**
  ```bash
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module list --catalog
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module install inbox
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module list
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module disable inbox
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module enable inbox
  docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module uninstall inbox --purge
  ```
- **Manual host mode (no Docker socket):** Set `ZOS_MODULE_HOST=manual`; `zos module install <slug>` prints container run instructions, and `zos module attach <slug> --url <mcp-url> [--ui-url <ui-url>]` finishes registration.
- **Upgrade an installed module:** Click **Upgrade** on `/admin/modules` (hot-reloads with zero restart) or run `zos module upgrade <slug>` from the CLI (restart `zos` after a CLI upgrade so the running router reconnects). Upgrades preserve the module's Postgres schema/data, `ZOS_MODULE_SECRET`, `providers` row, and owner tier/policy overrides.

---

## 11. Moving ZOS to a New Host

Verified end-to-end portability runbook (non-superuser Postgres role + encrypted backup restore + digest-pinned module image pull):

1. **Take a fresh encrypted backup on the old host:**
   ```bash
   ./deploy/backup/backup.sh
   ```
   Copy the newest `zos-backup-<ts>.dump.enc` file and your `.env` secrets (`ZOS_SECRET_KEY`, `ZOS_OWNER_PASSWORD_HASH`, `ZOS_BACKUP_KEY`, and optional `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`) securely to the new host.

2. **Prepare `.env` on the new host** (`chmod 600 .env`):
   - Set `ZOS_BASE_URL` to the new host's URL (`https://zos.<your-domain>` behind a tunnel, or `http://localhost:8787` locally).
   - Set a restricted non-superuser Postgres app role (recommended for all fresh deployments):
     ```env
     POSTGRES_USER=zos_boot
     POSTGRES_PASSWORD=<strong-bootstrap-password>
     POSTGRES_DB=zos
     ZOS_DB_USER=zos_app
     ZOS_DB_PASSWORD=<strong-app-password>
     ```
     On an empty Postgres volume, `deploy/postgres/01-init-app-role.sh` creates `zos_app` with `LOGIN CREATEROLE NOSUPERUSER NOCREATEDB`, transfers ownership of `zos` and schema `public` to `zos_app`, and installs `pg_trgm` in the shared `extensions` schema owned by `zos_app`.
   - *(On pre-M12 volumes where `pg_trgm` was originally installed in `public`, relocating `pg_trgm` into schema `extensions` requires Postgres superuser privileges once; `deploy/postgres/migrate-existing-to-app-role.sh` performs this automatically on bundled Postgres.)*
   - *(On hosted Postgres such as Neon, Fly, or RDS: create a database owned by a role with `CREATEROLE`, enable the trusted `pg_trgm` extension in schema `extensions`, and set `DATABASE_URL` directly.)*

3. **Start Postgres on the new host and restore the encrypted backup:**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml up -d --wait postgres
   ./deploy/backup/restore.sh /path/to/zos-backup-<ts>.dump.enc
   ```
   `restore.sh` decrypts the dump with `ZOS_BACKUP_KEY` and restores all core tables and module schemas (`mod_<slug>`) as `ZOS_DB_USER` with `--no-owner --no-privileges --no-comments`.

4. **Start ZOS (runs `zos migrate` automatically):**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml up -d --build zos
   ```

5. **Bring up installed container modules on the new host:**
   - For each installed module in `modules` (e.g. `inbox`), point its manifest in `catalog/modules/<slug>.toml` at a digest-pinned registry image (`image = "ghcr.io/zamborg/zos-<slug>@sha256:<64-hex>"`, or local `build = "modules/<slug>"`), then run **Upgrade** from `/admin/modules` (or via CLI followed by a `zos` restart):
     ```bash
     docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
       uv run --no-sync zos module upgrade inbox
     docker compose --env-file .env -f deploy/docker-compose.yml restart zos
     ```
   - `upgrade` pulls/builds the image, provisions the `mod_<slug>` Postgres role, transfers ownership of restored tables/sequences in `mod_<slug>` to `mod_<slug>`, preserves `ZOS_MODULE_SECRET` and owner tier/policy overrides, and starts the container on `zos_modules`.

6. **Verify the new host:**
   - `curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool` — confirm `ok: true` with `memory`, `inbox`, and (if `ZOS_SECRET_KEY` and Google creds match) `gcal` + `gmail` online.
   - Sign in at `/admin` and verify Overview, Approvals, Modules, MCP Servers, Clients, Tools & Policy, Activity, and `/admin/m/inbox/` all load cleanly.
   - *(Optional)* Point your Cloudflare Tunnel (or reverse proxy) and Google OAuth callback URL at the new host, and stop the old host (ZOS enforces single-instance startup via a Postgres advisory lock).

---

## 12. Running Automated Tests & Linting

Integration and end-to-end tests expect an isolated Postgres 17 test container on `127.0.0.1:55432` (never touching your live `deploy_zos_pgdata` volume):

```bash
docker run -d --name zos-pg-test \
  -e POSTGRES_USER=zos -e POSTGRES_PASSWORD=zos -e POSTGRES_DB=zos_test \
  -p 127.0.0.1:55432:5432 postgres:17
```

Run checks from `zos/`:

```bash
uv run ruff check
uv run pytest
(cd actuator && uv run ruff check && uv run pytest)
(cd modules/inbox && uv run ruff check && uv run pytest)
(cd modules/memory && uv run ruff check && uv run pytest)
(cd modules/google && uv run ruff check && uv run pytest)
```
