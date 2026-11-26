# Hosting ZOS

Self-contained deployment guide for running ZOS on a single Linux server (Ubuntu 24.04, ≥2 GB RAM, ≥20 GB disk). Written so a coding agent with SSH access can run the setup end-to-end while delegating only the minimal **👤 Human** steps below.

---

## Architecture at a Glance

- **Core (`zos`):** Control plane only — MCP endpoint (`/mcp`), OAuth 2.1 + DCR, policy engine, async approvals, audit log, AEAD vault, module lifecycle, and `/admin` UI + reverse proxy.
- **Database (`postgres:17`):** Runs inside Docker Compose under a restricted non-superuser app role (`zos_app`, `CREATEROLE`, `NOSUPERUSER`). Each stateful module receives its own isolated `mod_<slug>` role and schema.
- **Default modules:** `memory` (Postgres FTS + `pg_trgm` shared memory store) and `inbox` (async agent-to-agent and owner mailboxes), both running on the internal `zos_modules` network with zero internet egress.
- **Edge (`--edge`):**
  - `caddy` *(default)*: automatic Let's Encrypt HTTPS for `--domain` on ports `80`/`443`, proxying only `/mcp*`, `/.well-known/*`, `/oauth/*`, `/admin*`, `/connect/*`, and `/actuator` (WebSocket) to `zos:8080`, and returning `404` for everything else (including `/healthz`).
  - `cloudflared`: outbound Cloudflare Tunnel (no open inbound ports required).
  - `none`: binds `127.0.0.1:8787` only (for local machines or rehearsals).

---

## Step 1 — 👤 Provision a VPS & DNS Record

> **Human step:** An agent cannot open cloud accounts or modify your registrar.

1. **Rent an Ubuntu 24.04 VPS (≥2 GB RAM, 2 vCPU recommended).**
   - **Worked example (Hetzner Cloud):** Create a project → **Add Server** → Location of choice → Image **Ubuntu 24.04** → Type **CX22** (x86_64, 4 GB RAM) or **CAX11** (ARM64, 4 GB RAM) → add your SSH public key → **Create & Buy Now**.
   - Any other Ubuntu 24.04 provider (DigitalOcean, Vultr, Linode, AWS Lightsail) works identically.
   - Ensure inbound TCP ports **22** (SSH), **80**, and **443** are allowed in the provider firewall (only port 22 is needed if using `--edge cloudflared`).
2. **Add one DNS `A` record** at your DNS provider:
   - Type: `A`
   - Name: `zos` (for `zos.<your-domain>`)
   - Value: `<VPS_IPV4_ADDRESS>`
   - Wait until `dig +short zos.<your-domain>` returns `<VPS_IPV4_ADDRESS>`.
3. Hand the SSH connection (`root@<VPS_IPV4_ADDRESS>`) and domain name (`zos.<your-domain>`) to your agent (or run Steps 2–4 yourself).

---

## Step 2 — Run the Installer (Agent or Human)

SSH into the server as `root` and run `deploy/install.sh` pinned to the release tag (`zos-v3.0.0`):

```bash
curl -fsSL https://raw.githubusercontent.com/zamborg/lowercaseLabs/zos-v3.0.0/zos/deploy/install.sh -o /tmp/zos-install.sh
sudo bash /tmp/zos-install.sh \
  --domain zos.<your-domain> \
  --edge caddy \
  --ref zos-v3.0.0
```

*(To track the latest development branch instead of a release tag, use `main` in the URL and `--ref main`.)*

### What `install.sh` does (idempotent)

1. Installs Docker Engine and the Docker Compose plugin on Ubuntu 24.04 (if not already present).
2. Clones `https://github.com/zamborg/lowercaseLabs.git` into `/opt/zos` at `--ref` (or copies from `--local-source <path>` for offline/rehearsal installs).
   - **Prebuilt digest-pinned images (optional):** Set `ZOS_IMAGE_CORE` (e.g. `ghcr.io/zamborg/zos@sha256:<64-hex>`) and optionally `ZOS_IMAGE_MEMORY`, `ZOS_IMAGE_INBOX`, `ZOS_IMAGE_GMAIL`, `ZOS_IMAGE_GCAL` (`ghcr.io/zamborg/zos-<slug>@sha256:<64-hex>`) before running `install.sh` to pull published GHCR images by digest instead of building from source.
3. Generates strong random secrets (`POSTGRES_PASSWORD`, `ZOS_DB_PASSWORD`, `ZOS_SECRET_KEY`, `ZOS_BACKUP_KEY`) into `/opt/zos/zos/.env` (`chmod 600`) without ever printing them.
4. Starts Postgres 17, provisions the restricted non-superuser `zos_app` role (`CREATEROLE`, `NOSUPERUSER`) and shared `extensions` schema (`pg_trgm`), starts ZOS + Caddy (auto-HTTPS), runs `zos init`, and installs the `memory` and `inbox` modules.
5. Enables the nightly encrypted backup systemd timer (`zos-backup.timer` → `/var/backups/zos`).

> **Agent note:** When running `install.sh` on behalf of a human, pass `--skip-passphrase` during the automated install and then hand Step 3 directly to the human so the passphrase never passes through agent chat or logs.

---

## Step 3 — 👤 Set the Owner Passphrase

> **Human step:** Never paste your owner passphrase into an AI chat.

If you ran `install.sh` interactively, it prompts for the owner passphrase directly (hidden input, hashed with Argon2id into `/opt/zos/zos/.env`).

If an agent ran `install.sh --skip-passphrase`, SSH into the VPS yourself and run:

```bash
sudo bash /opt/zos/zos/deploy/install.sh --domain zos.<your-domain> --edge caddy
```

Because `install.sh` is idempotent, it skips already-completed steps, prompts only for the missing owner passphrase, hashes it into `.env`, and recreates `zos`.

---

## Step 4 — Verify the Deployment (Agent or Human)

Run from the VPS:

1. **Local health & default modules** (`/healthz` is intentionally local-only; public `https://zos.<your-domain>/healthz` returns `404`):
   ```bash
   curl -sf http://127.0.0.1:8787/healthz | python3 -m json.tool
   ```
   Confirm `"ok": true` and both `memory` (5 tools) and `inbox` (5 tools) report `"online": true`.
   When using `--edge caddy`, also confirm public `/healthz` is blocked (`404`):
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" https://zos.<your-domain>/healthz
   ```
2. **Non-superuser Postgres check:**
   ```bash
   cd /opt/zos/zos
   docker compose --env-file .env -f deploy/docker-compose.yml exec -T zos \
     uv run --no-sync python -c "import os, psycopg; print(psycopg.connect(os.environ['DATABASE_URL']).execute('SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user').fetchone())"
   ```
   Confirm `('zos_app', False)`.
3. **Admin UI:** Open `https://zos.<your-domain>/admin`, sign in with your owner passphrase, and verify Overview, Approvals, Modules, MCP Servers, Clients, Tools & Policy, Activity, `/admin/m/memory/`, and `/admin/m/inbox/` load cleanly.
4. **Nightly backup timer:**
   ```bash
   systemctl status zos-backup.timer
   sudo /opt/zos/zos/deploy/backup/backup.sh
   ```
   Confirm an encrypted `.dump.enc` file is created in `/var/backups/zos`.

---

## Step 5 — 👤 Connect Your Agents

> **Human step:** Point your MCP clients at `https://zos.<your-domain>/mcp`.

- **Claude Code:**
  ```bash
  claude mcp add --transport http zos https://zos.<your-domain>/mcp
  ```
- **Codex:**
  ```bash
  codex mcp add zos --url https://zos.<your-domain>/mcp
  ```
- **ChatGPT / Claude web & mobile:** Add a custom MCP server pointing at `https://zos.<your-domain>/mcp` and complete OAuth sign-in in your browser.
  - `chatgpt.com`, `claude.ai`, and `agent.meta.ai` are seeded on the OAuth redirect allowlist by default (loopback hosts are always allowed).
  - If a new external client is rejected during OAuth registration, open **`https://zos.<your-domain>/admin/clients`** — you will see `"<client> tried to connect from <host>"` with **Allow** and **Dismiss** buttons (both audited). Click **Allow** and retry connecting the client.
- **Headless / bearer scripts:** Create a bearer client on `/admin/clients` (or run `zos client add <name> --bearer`) and pass `Authorization: Bearer <token>`.

---

## Off-Box Nightly Backups (`rclone` Example)

`zos-backup.timer` runs `/opt/zos/zos/deploy/backup/backup.sh` nightly at `03:15 UTC` (±15m jitter), creating AES-256-CBC (PBKDF2) encrypted `pg_dump` archives in `/var/backups/zos` (retaining 14 backups locally).

To copy every encrypted backup off-box (S3, Cloudflare R2, Backblaze B2, Google Drive, etc.) without adding cloud SDKs to ZOS, configure `rclone` once and set `ZOS_BACKUP_POST_HOOK`:

1. Install and configure `rclone` as root:
   ```bash
   apt-get install -y rclone
   rclone config   # create a remote named e.g. 'offbox'
   ```
2. Add the hook to `/opt/zos/zos/.env`:
   ```env
   ZOS_BACKUP_POST_HOOK='rclone copy "$ZOS_BACKUP_FILE" offbox:zos-backups/'
   ```
3. Test immediately:
   ```bash
   sudo /opt/zos/zos/deploy/backup/backup.sh
   ```
4. **Store `ZOS_BACKUP_KEY` and `ZOS_SECRET_KEY` in your password manager.** Without `ZOS_BACKUP_KEY` the backup cannot be decrypted; without `ZOS_SECRET_KEY` encrypted vault credentials and module secrets cannot be decrypted after restore.

---

## Updating ZOS

`install.sh --update` takes an encrypted backup first, fetches the requested ref (or pulls updated digest images), runs database migrations, upgrades installed modules (preserving all schemas, secrets, and manual tier/policy overrides), and recreates containers:

```bash
sudo bash /opt/zos/zos/deploy/install.sh \
  --update \
  --domain zos.<your-domain> \
  --edge caddy \
  --ref zos-v3.0.0
```
*(Replace `zos-v3.0.0` with the new release tag when upgrading, or `--ref main` for the development branch.)*

---

## Moving an Existing ZOS to This Server

Follow the full migration and restore runbook in [TESTING.md § 11 (Moving ZOS to a New Host)](TESTING.md#11-moving-zos-to-a-new-host).

> **Note for pre-M12 Postgres volumes:** If you are upgrading an existing pre-M12 Postgres volume in place where `pg_trgm` was originally installed in schema `public`, relocating `pg_trgm` into the shared `extensions` schema requires Postgres superuser privileges once. On bundled Docker Compose Postgres, `deploy/postgres/migrate-existing-to-app-role.sh` handles this automatically (and `install.sh` runs it for you). On managed hosted Postgres without superuser, ensure `pg_trgm` is installed in schema `extensions` before installing the `memory` module.

---

## Edge Alternatives (`--edge cloudflared` / `--edge none`)

- **Cloudflare Named Tunnel (`--edge cloudflared`):**
  1. Create a Named Tunnel in Cloudflare Zero Trust pointing `zos.<your-domain>` to `http://zos:8080` (if running the bundled `cloudflared` container with `CLOUDFLARED_TOKEN` in `.env`) or `http://127.0.0.1:8787` (if running `cloudflared` on the host).
  2. Run `sudo bash /opt/zos/zos/deploy/install.sh --domain zos.<your-domain> --edge cloudflared`.
  3. Optionally add Cloudflare Access on browser-only paths (`/admin*`, `/oauth/authorize`, `/connect/*`) while leaving `/mcp`, `/.well-known/*`, `/oauth/register`, and `/oauth/token` bypassing Access.
- **Local / rehearsal (`--edge none`):**
  ```bash
  sudo bash deploy/install.sh --edge none --local-source .
  ```
  Binds only `127.0.0.1:8787` with no TLS proxy.

---

## Appendix — Optional Google Modules (`gmail` & `gcal`)

Gmail and Google Calendar run as two isolated egress modules (`gmail` and `gcal`) built from the shared `modules/google` image. Each module performs its own Google OAuth flow with narrower scopes and stores its refresh token encrypted inside its own Postgres schema (`mod_gmail` / `mod_gcal`).

1. **👤 Create a Google Cloud OAuth Web Client:**
   - Open [Google Cloud Console](https://console.cloud.google.com/), create or select a project, and enable **Gmail API** and **Google Calendar API** (plus **Google Calendar MCP** if you plan to use the `mcp` backend).
   - Under **Google Auth Platform → Audience**, choose **External** and publish (or add yourself as a Test User).
   - Under **Clients → Create Client → Web application**, add **both** Authorized Redirect URIs:
     - `https://zos.<your-domain>/admin/m/gmail/callback`
     - `https://zos.<your-domain>/admin/m/gcal/callback`
   - Copy the **Client ID** and **Client Secret**.
2. **Install the modules:**
   - Open **`https://zos.<your-domain>/admin/modules`**, click **Install** on **Gmail** and **Google Calendar**, and enter `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` for each (stored encrypted in the ZOS vault).
   - Or install via CLI:
     ```bash
     cd /opt/zos/zos
     docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
       uv run --no-sync zos module install gmail \
       --config GOOGLE_CLIENT_ID=<id> --config GOOGLE_CLIENT_SECRET=<secret>
     docker compose --env-file .env -f deploy/docker-compose.yml exec zos \
       uv run --no-sync zos module install gcal \
       --config GOOGLE_CLIENT_ID=<id> --config GOOGLE_CLIENT_SECRET=<secret>
     docker compose --env-file .env -f deploy/docker-compose.yml restart zos
     ```
3. **👤 Connect in your browser:**
   - Open `https://zos.<your-domain>/admin/m/gmail/` and click **Connect Gmail**.
   - Open `https://zos.<your-domain>/admin/m/gcal/` and click **Connect Google Calendar** (you can also switch between `api` REST v3 and `mcp` Calendar MCP backends on this page).
   - Verify `/healthz` reports `gmail` (3 tools) and `gcal` (9 tools on `api`) online.
