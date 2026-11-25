# ZOS V0 — Testing & Morning Setup Guide

ZOS V0 is **running live in OrbStack** right now at `http://localhost:8787` with the macOS actuator connected.

## What's running right now

- **ZOS + Postgres 17 (OrbStack Docker Compose):** `deploy/docker-compose.yml`
  - Bound to `127.0.0.1:8787` only (`restart: unless-stopped`, persistent volume `deploy_zos_pgdata`).
  - Principal `zubin`, space `main`, hosted `memory` provider, and all 14 V0 `tool_tiers` are initialized.
- **Mac actuator (LaunchAgents):**
  - `com.zos.tmux` (`tmux -L zos` isolated from actuator permissions)
  - `com.zos.actuator` (`~/.zos-actuator/venv`, dialed into `ws://127.0.0.1:8787/actuator`)
- **Quick health check:**
  ```bash
  curl -s http://127.0.0.1:8787/healthz | python3 -m json.tool
  ```
  Expect `memory` (`online: true`), `mac` (`online: true`), and `gcal`/`gmail` (`online: false` until you connect Google below).
- **Owner admin dashboard:** open **`http://localhost:8787/admin`** and sign in with the passphrase in `zos/.owner-passphrase`.
  Use **Overview** for live provider status and Google connect, **Clients** to mint or revoke bearer/OAuth tokens, **Tools & Policy** to set or clear per-client tool overrides (replacing `zos policy set`), and **Activity** to inspect the cursor-paginated audit trail.

## Local secrets (gitignored, mode `0600`)

- `zos/.env` — Postgres password, `ZOS_SECRET_KEY`, and single-quoted `ZOS_OWNER_PASSWORD_HASH`.
- `zos/.owner-passphrase` — generated owner passphrase for `/oauth/authorize` and `/connect/google`. Read with `cat zos/.owner-passphrase` (you can delete it once stored in your password manager, or rotate by putting a new single-quoted hash from `uv run zos hash-password` into `zos/.env`).
- `~/.zos-actuator/config.toml` — actuator WebSocket URL and token.

---

## 1. Connect Claude Code and Codex

### Option A — OAuth 2.1 (browser login)

Point both clients at **`http://localhost:8787/mcp`**:

- **Claude Code:**
  ```bash
  claude mcp add --transport http zos http://localhost:8787/mcp
  ```
  When prompted to sign in, your browser opens `http://localhost:8787/oauth/authorize` — enter the passphrase from `zos/.owner-passphrase` and click **Approve**.
- **Codex:**
  ```bash
  codex mcp add zos --url http://localhost:8787/mcp
  ```
  (Or configure `[mcp_servers.zos]` with `url = "http://localhost:8787/mcp"` in `~/.codex/config.toml` and complete the OAuth login.)

### Option B — Bearer token fallback

If either client prefers a static bearer header over browser OAuth:

```bash
cd /Users/zubinaysola/Documents/personal/lowercaseLabs/zos
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos client add claude-code --bearer
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos client add codex --bearer
```

Pass `Authorization: Bearer <token>` to `http://localhost:8787/mcp`.

### Smoke tests to try

1. **Shared memory (marquee problem):** Ask Claude Code to store a memory (`memory_put`), then ask Codex to retrieve or search it (`memory_get` / `memory_search`).
2. **Status & Mac shell:** Call `zos_status`, then `mac_shell_run` (external tier — clients with MCP elicitation prompt for approval; clients without elicitation receive the exact `zos policy set` override command).
3. **Granting an override (if needed):**
   ```bash
   docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
     uv run --no-sync zos policy set --client <client-name-or-id> mac_shell_run allow
   ```
4. **Actuator live visibility:** Stop and restart the actuator while connected — `mac_*` tools vanish and reappear automatically via `notifications/tools/list_changed`:
   ```bash
   launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zos.actuator.plist
   ```

---

## 2. Pending morning steps (require GUI / browser)

### A. Grant Full Disk Access for `mac_messages_search`

1. Open **System Settings → Privacy & Security → Full Disk Access**.
2. Click **`+`**, press **`Cmd+Shift+G`**, and select the exact resolved Python binary:
   ```text
   /Users/zubinaysola/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python3.12
   ```
3. Restart the actuator so it picks up the permission:
   ```bash
   launchctl kickstart -k gui/$(id -u)/com.zos.actuator
   ```
4. Test `mac_messages_search` from any connected MCP client. *(Re-grant Full Disk Access if you upgrade Python.)*

### B. Connect Google Calendar (MCP) + Gmail (REST)

1. In [Google Cloud Console](https://console.cloud.google.com/), create (or pick) a project (e.g. `zos-personal`) and enable:
   - **Gmail API** (`gmail.googleapis.com`)
   - **Google Calendar MCP** (`calendarmcp.googleapis.com`) — and optionally **Google Calendar API** (`calendar-json.googleapis.com`) if you ever want to switch `gcal` to `backend: api` fallback.
2. Under **Google Auth Platform → Audience**: set User Type **External**, then click **Publish App** (**In production**; do *not* submit for verification).
   - *Fallback if restricted Gmail scopes are blocked on an unverified personal account:* switch to **Testing** mode, add your personal Gmail as a Test User, and note the 7-day refresh token expiry.
3. Under **Google Auth Platform → Data Access**, add the unioned Calendar MCP + Gmail scopes:
   - `https://www.googleapis.com/auth/calendar.calendarlist.readonly`
   - `https://www.googleapis.com/auth/calendar.events.freebusy`
   - `https://www.googleapis.com/auth/calendar.events.readonly`
   - `https://www.googleapis.com/auth/calendar.events`
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
4. Under **Clients → Create Client → Web application**:
   - Tick **"Use this client for an AI-powered agent"** (required by Google's Calendar MCP server).
   - Add Authorized Redirect URI:
     ```text
     http://localhost:8787/connect/google/callback
     ```
     *(Optional: also add `http://127.0.0.1:8765/callback` if you want to run `uv run python spikes/google/oauth_spike.py`.)*
5. Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to `zos/.env`, then recreate the container:
   ```bash
   cd /Users/zubinaysola/Documents/personal/lowercaseLabs/zos
   docker compose --env-file .env -f deploy/docker-compose.yml up -d --force-recreate zos
   ```
6. Open **`http://localhost:8787/connect/google`** in your browser, enter the passphrase from `zos/.owner-passphrase`, and complete Google consent. Both `gcal_*` (proxied from `https://calendarmcp.googleapis.com/mcp/v1`) and `gmail_*` (REST adapter) appear immediately via `notifications/tools/list_changed`.
   - *Backend switch:* toggle `gcal` between `api` and `mcp` live with zero restart from **`/admin`** (Overview → Providers), or via CLI `zos provider set-backend gcal api|mcp` (persisted in `providers.config` and audited as `provider.backend_changed`).

### C. Public exposure via Cloudflare Named Tunnel (for ChatGPT & remote MCP clients)

When ready to expose ZOS at `https://zos.<your-domain>` (e.g. so ChatGPT connectors can reach `/mcp`):

1. **Configure ZOS environment (`zos/.env`):**
   - Set `ZOS_BASE_URL=https://zos.<your-domain>` (turns on `Secure` cookies on `/admin`, sets OAuth `issuer`/`resource`/PRM/`WWW-Authenticate` to the public URL, and sets the Google callback to `https://zos.<your-domain>/connect/google/callback`).
   - Set `ZOS_ALLOWED_REDIRECT_HOSTS=chatgpt.com` (default is empty = loopback-only; ChatGPT MCP connectors use `https://chatgpt.com/connector_platform_oauth_redirect` and `https://chatgpt.com/connector/oauth/<id>`).
   - Recreate the container:
     ```bash
     cd /Users/zubinaysola/Documents/personal/lowercaseLabs/zos
     docker compose --env-file .env -f deploy/docker-compose.yml up -d --force-recreate zos
     ```
2. **Update Google Cloud Console redirect URI:**
   - Under **Google Auth Platform → Clients → your Web client → Authorized redirect URIs**, add:
     ```text
     https://zos.<your-domain>/connect/google/callback
     ```
3. **Re-point local MCP clients (`Claude Code` & `Codex`):**
   - OAuth access tokens are bound to `ZOS_BASE_URL` (RFC 8707 `resource`), so once `ZOS_BASE_URL` switches to `https://zos.<your-domain>`, re-point Claude Code and Codex to **`https://zos.<your-domain>/mcp`** and sign in again.
   - **Mac actuator stays local:** keep `~/.zos-actuator/config.toml` dialing `ws://127.0.0.1:8787/actuator` directly — `/actuator` is intentionally blocked on the public tunnel.
4. **Cloudflare Tunnel ingress allowlist (`deploy/cloudflared/config.yml.example`):**
   - Copy `deploy/cloudflared/config.yml.example` to `~/.cloudflared/config.yml` and fill in your tunnel UUID, home path, and hostname (`zos.<your-domain>`).
   - Only `/mcp`, `/.well-known/*`, `/oauth/*`, `/admin*`, and `/connect/*` are routed to `http://localhost:8787`; everything else (including `/actuator` and `/healthz`) returns `404`.
   - Use `deploy/launchd/com.zos.cloudflared.plist` (substituting `{{HOME}}` and `{{TUNNEL_NAME}}`) as a LaunchAgent template for `cloudflared tunnel run`.
5. **Cloudflare Access (Zero Trust) on browser-only paths:**
   - Put a **Cloudflare Access** self-hosted application policy (allowing only your email) on the browser-only owner paths:
     - `zos.<your-domain>/oauth/authorize`
     - `zos.<your-domain>/admin*`
     - `zos.<your-domain>/connect/*`
   - Leave the machine-to-machine OAuth & MCP paths **public** (bypass Access) so ChatGPT, Claude Code, and Codex can discover, register, exchange tokens, and call MCP:
     - `zos.<your-domain>/mcp`
     - `zos.<your-domain>/.well-known/*`
     - `zos.<your-domain>/oauth/register`
     - `zos.<your-domain>/oauth/token`
6. **Abuse protections active automatically:**
   - One global passphrase rate limiter (5 failures/min, 20/hour, exponential backoff) protects `/admin/login`, `/oauth/authorize`, and `/connect/google` together, auditing each failure as `auth.failed` (`outcome=denied`, zero passphrase material).
   - `/oauth/register` is globally rate-limited (10/hour) and capped at 50 total OAuth clients; prune unused OAuth clients anytime in **`/admin/clients`** (or `zos client delete <client>`).

### D. Later / optional

- **Cloud deploy (Fly + Neon):** `deploy/fly.toml` and `deploy/Dockerfile` are ready whenever you want to deploy off localhost.

---

## 3. Honest V0 Check & Definition-of-Done Audit

| Milestone / Item | Automated / Local Status | Still Pending (👤 Browser / GUI / Cloud) |
|---|---|---|
| **M0 — Spikes** | Local hosting spike (`ping`/`toggle`/`list_changed`/idle) passed; `DECISIONS.md` updated. | Fly.io 120s idle spike (deferred for OrbStack); M0.3 live Google personal-account restricted-scope spike (`spikes/google/oauth_spike.py` ready). |
| **M1 — Core, Policy, Memory, Deploy** | All unit/integration/e2e tests pass (including `untrusted` ceiling, naming truncate+hash, annotations, elicitation, audit); OrbStack live stack running on `127.0.0.1:8787`; Codex CLI (`gpt-6.1-terra`) bearer smoke passed. | Fly + Neon cloud deploy smoke (deferred); manual Claude Code + Codex shared-memory smoke. |
| **M2 — OAuth 2.1 AS** | Full MCP SDK `OAuthClientProvider` e2e passes locally and live against OrbStack; PKCE, single-use/expired codes, DCR rules, refresh rotation & family reuse isolation all tested. | Manual browser OAuth sign-in from Claude Code and Codex. |
| **M3 — Mac Actuator** | Fixture `chat.db`, `typedstream` (127/128/70k/UTF-8/`NSMutableString`/truncated), real `tmux`, and core+actuator e2e (`list_changed`, `ACTUATOR_OFFLINE`, `ACTUATOR_DISCONNECTED_MID_CALL`, replacement, auth rejection) all pass; live LaunchAgent stop/restart tool visibility + live `mac_shell_run` via Codex passed. | Grant Full Disk Access to the resolved Python binary and run real-device `mac_messages_search` count-only smoke. |
| **M4 — Google Upstream** | Vault AES-GCM + concurrent refresh lock + `respx`-mocked `/connect/google`, Calendar REST fallback (`backend=api`), Gmail bounded metadata search/read/MIME threading, 429 backoff, 401 refresh, `gmail_send` elicitation, and revocation all pass. | Google Cloud Web OAuth client setup, live `/connect/google` consent, and live Calendar/Gmail/`gmail_send` elicitation smoke. |
| **M5 — Upstream MCP Proxy (`gcal`)** | Generic `McpUpstreamProvider` verified against in-process Streamable HTTP fake upstream: prefixing, verbatim call + `structuredContent`/`isError` passthrough, upstream→downstream `list_changed`, upstream form elicitation passthrough + decline + audit, 401→refresh→retry & reconnect error, upstream down→offline, unknown preview tool→`external`. Live `/healthz` reports `gcal` with `backend: mcp`. | Live smoke against Google's `calendarmcp.googleapis.com/mcp/v1` (counts/shapes only) after Zubin connects Google. |
| **M6 — Owner Admin Dashboard (`/admin`)** | Shared `AdminService` layer backs both CLI and `/admin`; auth (Argon2 login, signed HttpOnly `SameSite=Strict` 12h cookie, `Secure` on HTTPS, expiry, rate limiting, CSRF 403, security headers), all 4 pages (Overview, Clients, Tools & Policy matrix, cursor-paginated Activity with digests only), and CLI audit parity all tested; live `/admin` login verified on OrbStack. | None — ready to use at `http://localhost:8787/admin`. |
| **M7 — Pre-Public Hardening** | Shared global passphrase limiter (5/min, 20/hr, backoff) + `auth.failed` audit across `/admin/login`, `/oauth/authorize`, `/connect/google`; `ZOS_ALLOWED_REDIRECT_HOSTS` loopback-only default + `chatgpt.com` allowlist + prominent redirect host on approve page; DCR 10/hr limit + 50 OAuth client cap + `/admin/clients` pruning; public `https://` base URL test; `PublicSchemeMiddleware` HTTPS trailing-slash redirects; declarative `deploy/cloudflared/config.yml.example` + `com.zos.cloudflared.plist`. | Cloudflare named tunnel live at `https://zos.aysola.org`. |
| **M8 — Context Layer & Baseline** | `zos memory import` (`claude-code` & `codex` importers, JSONL stdin host wrapper `deploy/scripts/import-memories.sh`, shared `MemoryService`, idempotent & audited); MCP `InitializeResult.instructions` (+ `ZOS_INSTRUCTIONS_FILE` override); `gcal` REST parity (`get_event`, `list_calendars`, `search_events`, `freebusy`, `respond_to_event` + `gcal_freebusy` read tier); `memory_list` (cursor-paginated, no content) + `author`/`created_at`/`updated_at` on `memory_get`/`search`; actuator AddressBook contact names & name-to-handle lookup in `mac_messages_search`; nightly encrypted `pg_dump` backup (`deploy/backup/{backup,restore}.sh`, `com.zos.backup.plist`, `/admin` Overview last-backup status, verified live restore into `zos-pg-test`). | Re-enable iCloud Drive on macOS if `~/Library/Mobile Documents/com~apple~CloudDocs/zos-backups/` should sync off-device automatically (directory created locally). |
| **Definition of Done (V0)** | Every call audited in `events`; `DECISIONS.md` current; zero `later/` scope built; all Docker ports bound strictly to `127.0.0.1`. | End-to-end human session with both Claude Code and Codex signed in via OAuth using shared memory + FDA Messages + live Google. |

---

## 4. M8 Context & Operations Quick Commands

- **Import host agent memories into ZOS (idempotent, prints counts only):**
  ```bash
  cd /Users/zubinaysola/Documents/personal/lowercaseLabs/zos
  ./deploy/scripts/import-memories.sh
  ```
- **Run an encrypted backup manually:**
  ```bash
  ./deploy/backup/backup.sh
  ```
  Encrypted dumps are written to `~/Library/Mobile Documents/com~apple~CloudDocs/zos-backups/` (retaining the 14 newest) and recorded for `/admin` Overview.
- **Restore a backup into a target Postgres container:**
  ```bash
  ./deploy/backup/restore.sh ~/Library/Mobile\ Documents/com~apple~CloudDocs/zos-backups/zos-backup-<ts>.dump.enc --container zos-pg-test --db zos_restore_test --user postgres
  ```

---

## Running the automated test suites

From `zos/`:

```bash
uv run pytest
uv run ruff check
(cd actuator && uv run pytest && uv run ruff check)
```

*(Integration/E2E tests target the local `zos-pg-test` Postgres 17 container on `127.0.0.1:55432`, completely isolated from the live `deploy_zos_pgdata` volume.)*
