# ZOS — Architectural & Implementation Decisions

Log any choice not settled by `PLAN.md`, `SCHEMA.md`, or `BUILD.md`, plus spike findings and any deviations.

Format: `## YYYY-MM-DD — <Short title>` with **Decision**, **Why**, and **Impact**.

---

## 2026-11-23 — Python package and dependency baseline (Task 0.1)

- **Decision:** Set `requires-python = ">=3.12"` (with `.python-version` pinned to `3.12`) with `starlette` + `uvicorn[standard]` as the ASGI stack, `psycopg[binary,pool]` for Postgres 3 async pooling, `argon2-cffi` for owner passphrase hashing, and `hatchling` as build backend.
- **Why:** Matches `BUILD.md` stack (Python 3.12, Starlette/FastAPI, official `mcp` SDK, psycopg 3, Typer, cryptography, httpx). Starlette avoids redundant FastAPI overhead since the MCP Streamable HTTP transport and minimal HTML forms mount cleanly as Starlette ASGI routes.
- **Impact:** Both `zos` and `zos-actuator` target Python 3.12 via `uv`.

## 2026-11-23 — `mcp==2.3.0` low-level Server & Streamable HTTP APIs (Task 0.2)

- **Decision:** Subclass `mcp.server.lowlevel.Server` to override `create_initialization_options()` with `NotificationOptions(tools_changed=True)`, register `on_list_tools` / `on_call_tool` via constructor kwargs, and track live `Connection` instances via the `notifications/initialized` handler + `conn.exit_stack.callback(...)` for broadcasting `conn.send_tool_list_changed()`.
- **Why:** Verified against installed `mcp==2.3.0` source: `StreamableHTTPSessionManager` calls `serve_loop` with `init_options=None`, which defaults `tools_changed=False`; `EventSourceResponse` from `sse-starlette` already emits SSE keepalive comments every 15s (`DEFAULT_PING_INTERVAL = 15`); and `Server.streamable_http_app()` auto-enables localhost-only DNS rebinding protection unless `host="0.0.0.0"` or explicit `TransportSecuritySettings` is passed.
- **Impact:** Works cleanly for both localhost and containerized/proxied Streamable HTTP deployments, and enables dynamic actuator tool updates in M3.

## 2026-11-23 — Hosting target shifted to OrbStack Docker Compose; Fly & Google spikes deferred (Tasks 0.2, 0.3)

- **Decision:** Host V0 locally on OrbStack via `deploy/docker-compose.yml` (`zos` + `postgres:17` with named volume, bound to `127.0.0.1` only, `restart: unless-stopped`, `ZOS_BASE_URL=http://localhost:<port>`). Defer Fly.io hosting spike and Fly deployment (`fly.toml` kept declarative in `deploy/`). Defer M0.3 live Google OAuth consent spike to Zubin's morning browser step (`spikes/google/oauth_spike.py` is ready to run; M4 will be built and tested against `respx` mocks).
- **Why:** Per Zubin's overnight direction: wake up to V0 live on OrbStack without cloud spend or blocking on browser-only Google Console / OAuth consent steps. Local hosting spike (`spikes/hosting/server.py` + `client.py`) passed connect, `ping`, `toggle`, `notifications/tools/list_changed`, and post-idle calls.
- **Impact:** Self-host Docker Compose stack is first-class; Fly deployment and live Google consent remain ready whenever promoted to cloud.

## 2026-11-23 — Schema implementation details in `0001_init.sql` (Task 1.1)

- **Decision:** Added nullable `resource TEXT` columns to `oauth_codes` and `tokens` (for RFC 8707 audience binding in M2.3), `created_at TIMESTAMPTZ NOT NULL DEFAULT now()` to `tokens`, and a surrogate `id UUID PRIMARY KEY DEFAULT gen_random_uuid()` to `client_policies` (preserving `UNIQUE NULLS NOT DISTINCT (principal_id, client_id, tool_pattern)`).
- **Why:** Supports RFC 8707 resource-indicator verification cleanly without a separate migration and makes token/policy rows straightforward to inspect and revoke.
- **Impact:** Strictly additive over `SCHEMA.md`; all constraints and semantics match `SCHEMA.md` verbatim.

## 2026-11-23 — Custom minimal OAuth 2.1 AS routes over `mcp.server.auth.routes` (Tasks 2.1–2.4)

- **Decision:** Implement discovery (`/.well-known/oauth-protected-resource`, `/.well-known/oauth-protected-resource/mcp`, `/.well-known/oauth-authorization-server`), DCR (`/oauth/register`), owner-passphrase approval (`/oauth/authorize`), and token exchange/refresh (`/oauth/token`) directly in `src/zos/auth/oauth.py` rather than mounting `mcp.server.auth.routes.create_auth_routes`.
- **Why:** Verified against installed `mcp==2.3.0`: `mcp.server.auth.routes.build_metadata` hardcodes `token_endpoint_auth_methods_supported=["client_secret_post", "client_secret_basic"]` (omitting `"none"` required for public PKCE clients) and uses `/authorize`/`/token`/`/register` rather than `/oauth/*`. Implementing directly keeps our single-user AS minimal, supports `token_endpoint_auth_method="none"`, enforces RFC 9207 `iss` and RFC 8707 `resource` binding, and integrates cleanly with our `family_id`/`rotated_at` refresh-token reuse revocation.
- **Impact:** Verified end-to-end against `mcp.client.auth.OAuthClientProvider` in `tests/e2e/test_m2_oauth.py`.

## 2026-11-23 — Dedicated `tmux -L zos` spawns `/bin/zsh -f` with explicit launchd PATH/HOME (Task 3.4/3.5)

- **Decision:** Launch the persistent `tmux -L zos` pane with `-f /dev/null` and `/bin/zsh -f` (no interactive `.zshrc`), setting `HOME`, `PATH` (`/opt/homebrew/bin:...`), `TERM`, `LANG`, `WorkingDirectory`, and `AbandonProcessGroup = true` in `com.zos.tmux.plist`.
- **Why:** Observed during live macOS launchd smoke testing: interactive login `-zsh` hangs indefinitely in a detached tmux pane with no attached client (interactive prompt/plugin init stalls), whereas `/bin/zsh -f` is ready in <50ms, deterministic, and still preserves cwd and exported env vars across `mac_shell_run` calls.
- **Impact:** `mac_shell_run` executes reliably under launchd without inheriting user `.zshrc` or `.tmux.conf` side effects.

## 2026-11-24 — Generic `McpUpstreamProvider` and `gcal` Calendar MCP wiring (M5)

- **Decision:** Implement `McpUpstreamProvider` generically over `UpstreamTokenSource` (vault-backed) using `mcp`'s Streamable HTTP client with an `httpx2.Auth` bearer handler (one automatic refresh+retry on 401), serialize `call_tool` per upstream provider via `asyncio.Lock` to correlate upstream `elicitation/create` 1:1 with the exact downstream client & `request_id` (audited as `elicitation.passthrough`), default `gcal` to `backend: mcp` (`https://calendarmcp.googleapis.com/mcp/v1`) with `backend: api` selectable via `providers.config` or `ZOS_GCAL_BACKEND`, and surface `"backend"` on upstream providers in `/healthz`.
- **Why:** Keeps `ToolProvider` untouched while preventing cross-client elicitation misrouting across `ClientSession`'s shared receive loop, keeps policy/tiers strictly under ZOS `tool_tiers` (unknown preview tools default to `external`), and makes the active backend visible on `/healthz` without breaking its existing shape.
- **Impact:** `gcal` exercises remote MCP proxying while `gmail` stays on our REST adapter, both sharing one Google credential and unioned scopes (`calendar.*` + `gmail.*`).

## 2026-11-24 — Shared `AdminService` layer and stateless HMAC `/admin` cookies (M6)

- **Decision:** Extract all administrative mutations and audit recording into `src/zos/admin_service.py` (`AdminService`) shared by both `src/zos/cli.py` and `/admin`; authenticate `/admin` via stateless HMAC-SHA256 signed cookies (`zos_admin_session`, `HttpOnly`, `SameSite=Strict`, `Path=/admin`, 12h TTL, `Secure` when `ZOS_BASE_URL` is `https://`) carrying a per-session CSRF token, with in-memory per-IP login failure rate limiting (5 failures / 60s → 429) and strict CSP/clickjacking headers.
- **Why:** Guarantees CLI and `/admin` never drift in SQL or audit events, avoids adding a separate `admin_sessions` DB table (no schema migration required), keeps `/admin` zero-build/zero-JS-framework HTML, and enforces CSRF + brute-force protection out of the box.
- **Impact:** Every `/admin` mutation emits the identical `events` records as `zos` CLI commands; `zos policy clear` was also exposed on the CLI to mirror `/admin/policy/clear`.

## 2026-11-24 — Pre-public hardening for Cloudflare Tunnel exposure (M7)

- **Decision:** Replace per-IP admin login limiting with one shared global `GlobalPassphraseRateLimiter` (5 failures/min, 20 failures/hour, exponential backoff starting at 10s, auditing each failed verification as `auth.failed` / `outcome='denied'` with zero passphrase material) across `/admin/login`, `/oauth/authorize`, and `/connect/google`. Restrict unauthenticated DCR (`/oauth/register`) to loopback hosts (`localhost`, `127.0.0.1`, `::1`, `http`/`https`) plus explicit HTTPS hosts in `ZOS_ALLOWED_REDIRECT_HOSTS` (default empty = loopback only; rejects custom schemes, fragments, userinfo, and unlisted hosts with `invalid_redirect_uri`), render the redirect host prominently on `/oauth/authorize`, rate-limit DCR globally (default 10/hour via `ZOS_DCR_MAX_PER_HOUR`), cap total OAuth clients (default 50 via `ZOS_MAX_OAUTH_CLIENTS`), and add OAuth client pruning (`POST /admin/clients/delete` + `zos client delete`, audited as `client.deleted`). Ship declarative `deploy/cloudflared/config.yml.example` (allowlisting only `/mcp`, `/.well-known/*`, `/oauth/*`, `/admin*`, `/connect/*`, returning 404 for `/actuator`, `/healthz`, and all other paths) and `deploy/launchd/com.zos.cloudflared.plist`.
- **Why:** Behind `cloudflared`, all proxied traffic shares one local peer IP so per-IP bucketing fails; single-owner ZOS benefits from global lockout and shared auditing across all passphrase forms. OpenAI's official ChatGPT MCP connector redirect URIs (`https://chatgpt.com/connector_platform_oauth_redirect` and `https://chatgpt.com/connector/oauth/<id>`) live on `chatgpt.com`, so `ZOS_ALLOWED_REDIRECT_HOSTS=chatgpt.com` tightly scopes external DCR while keeping loopback CLI clients working out of the box.
- **Impact:** Localhost deployments work unchanged with loopback-only defaults; promoting to `https://zos.<domain>` requires setting `ZOS_BASE_URL`, `ZOS_ALLOWED_REDIRECT_HOSTS=chatgpt.com`, adding the HTTPS Google callback URI, and re-pointing Claude Code/Codex to the public `/mcp` URL while keeping the local Mac actuator on `ws://127.0.0.1:8787/actuator`.

## 2026-11-24 — Context layer, memory import transport, and encrypted backups (M8)

- **Decision:**
  1. **Memory import transport:** Split host discovery (`zos memory export <claude-code|codex>`) from DB ingestion (`zos memory import -` reading JSONL over stdin), wrapped by `deploy/scripts/import-memories.sh` piping into `docker compose exec -T zos`. Both `memory_put` and `zos memory import` share `src/zos/memory_service.py` (`MemoryService.put_memory`), attributing imports to a dedicated `memory-import` client and auditing creates/updates as `memory.imported` (digests only, never raw content).
  2. **MCP server `instructions`:** Provide spec-native `InitializeResult.instructions` (<1200 chars, client-agnostic) via `src/zos/instructions.py`, overridable via `ZOS_INSTRUCTIONS_FILE`.
  3. **Tier seed migrations:** Ship additive idempotent SQL migrations (`0002_m8_gcal_freebusy_tier.sql`, `0003_m8_memory_list_tier.sql`) alongside `V0_TIER_SEEDS` so existing live Postgres databases receive new read-tier seeds automatically on startup.
  4. **Encrypted backups & `/admin` Overview status:** Pin `/usr/bin/openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:ZOS_BACKUP_KEY` (verified on macOS LibreSSL 3.3.6) so backup and restore always use identical PBKDF2 defaults without exposing keys in `ps`. Record backup metadata via `zos backup record` into `events` (`backup.completed`) so the containerized `/admin` Overview displays last backup time & size with no extra host bind-mounts.
- **Why:** Keeps container filesystem isolation intact (the container never mounts `~/.claude` or iCloud), guarantees backup/restore compatibility across macOS LibreSSL vs Homebrew OpenSSL, and avoids manual DB edits for new tool tier seeds.
- **Impact:** Running `./deploy/scripts/import-memories.sh` and `./deploy/backup/backup.sh` works cleanly on the host against the live OrbStack stack; `~/Library/Mobile Documents/com~apple~CloudDocs/zos-backups/` was created locally (enable macOS iCloud Drive if off-device sync is desired).

## 2026-11-24 — Owner-managed routed remote MCP servers (M9a)

- **Decision:** Generalize `credentials.issuer` CHECK constraint (`0004_m9a_generic_upstream_credentials.sql`) to `length(trim(issuer)) > 0` so any upstream or future V2 module can store AEAD-encrypted secrets (`mcp_bearer`, `mcp_oauth`) in `credentials`. Store generic routed MCP registrations in `providers` (`kind='upstream'`) with `config = {runtime: "remote", transport: "mcp", url, auth_mode, default_tier}` (aligning with `V2.md`'s `runtime: remote` module manifest shape; zero secrets in `config`). Drive OAuth upstreams through the installed `mcp.client.auth.OAuthClientProvider` (`VaultOAuthTokenStorage` implementing `TokenStorage` over the AEAD vault) behind the existing owner passphrase-gated `/connect/<slug>` → `/connect/<slug>/callback` flow (mobile-friendly, shared global rate limiter), persisting discovered `OAuthMetadata`/`ProtectedResourceMetadata` so token refresh works across restarts and failed/revoked refreshes transition to `needs_reconnect` with a clear reconnect URL. Enforce strict slug format `^[a-z][a-z0-9]{1,15}$` (rejecting `zos` and built-ins), sanitize upstream tool names to `[a-zA-Z0-9_-]+` while forwarding raw names upstream, and reject exposed-name collisions. Implement `--default-tier` via a `<slug>_*` row in `tool_tiers` (so per-tool overrides set on `/admin/policy/tier` automatically win by longest-literal specificity).
- **Why:** Reuses `McpUpstreamProvider`, `CredentialVault`, `tool_tiers`, and the Access-gated `/connect/` path without inventing parallel auth or policy machinery, while directly matching `V2.md`'s future remote-module manifest shape.
- **Impact:** Any Streamable HTTP remote MCP server (`none`, `bearer`, or `oauth`) can be added, configured, connected from a phone, tiered per tool, disconnected, and removed live with zero restarts via `zos upstream ...` or `/admin/upstreams`.

## 2026-11-24 — Remote MCP connector catalog & V1 manifest format (M9b)

- **Decision:** Check in 20 empirically verified remote connector manifests under `catalog/connectors/<slug>.toml` (`manifest_version = 1`, `runtime.kind = "remote"`, validated by Pydantic in `src/zos/catalog.py`; malformed entries are logged and skipped without breaking startup). Nothing is installed by default. On install (`zos catalog install <slug> [--as <alias>]` or `/admin/upstreams/install`), persist `manifest_tiers` and `managed_tier_patterns` in `providers.config` and resolve glob/exact patterns into `tool_tiers` on initial connect and on `notifications/tools/list_changed` without ever overwriting owner-edited tiers; remove manifest-managed tiers on uninstall.
- **Servers without Dynamic Client Registration (DCR) verified 2026-11-24:**
  - **GitHub (`https://api.githubcopilot.com/mcp/`):** `https://github.com/login/oauth` does not advertise `registration_endpoint` (no DCR). Modeled as `auth.mode = "bearer"` with `token_help` explaining fine-grained PAT minting and minimum scopes.
  - **Slack (`https://mcp.slack.com/mcp`):** `https://mcp.slack.com` PRM/AS metadata has no `registration_endpoint` (no DCR). Modeled as `auth.mode = "oauth"` with `requires_client_registration = true` and `/admin/upstreams/oauth-client` fields to store a pre-registered `client_id` + `client_secret` in the AEAD vault before clicking Connect.
  - **Asana (`https://mcp.asana.com/v2/mcp`; legacy `/sse` returns 410 Gone):** `https://app.asana.com` has no `registration_endpoint` (no DCR). Modeled identically as `auth.mode = "oauth"` with `requires_client_registration = true`.
- **Why:** Matches `V2.md`'s future module manifest (`runtime: remote`) while honoring real-world OAuth server capabilities empirically verified against live PRM/ASM metadata, and keeps owner hand-tuned tool tiers safe across upstream tool list changes.
- **Impact:** All 20 verified connectors appear in `/admin/upstreams` ("Available Connectors") and `zos catalog list`, support multi-instance installs via `--as`, and remain uninstalled until explicitly added by the owner.

## 2026-11-24 — Async approvals behind `require_approval` for non-elicitation clients (M9c)

- **Decision:** Keep MCP `elicitation/create` unchanged whenever `session_supports_elicitation(session)` is true; otherwise route `require_approval` calls through `ApprovalManager` (`src/zos/policy/approvals.py`, backed by `pending_approvals` in `0005_m9c_pending_approvals.sql`).
  1. **Encrypted-at-rest arguments & immediate purge:** Encrypt pending tool arguments with the existing AES-256-GCM `CredentialVault` (AAD bound to `(principal_id, "pending_approval", approval_id, key_version)`). Never store plaintext arguments in Postgres or `events`. Immediately purge `args_ciphertext` and `args_nonce` to `NULL` when decided (`approved`/`denied`), consumed, or expired (with a sweep on every access).
  2. **Hold timeout (`ZOS_APPROVAL_WAIT_S=45`):** Hold the tool call up to 45 seconds waiting on an in-process `asyncio.Event` (with a 1s DB fallback check). `45s` stays safely below observed transport/proxy timeouts: uvicorn `timeout_keep_alive=75s`, `cloudflared` origin timeout (`90s`), Cloudflare proxy timeout (`100s`), MCP SDK Streamable HTTP read timeout (`300s`), and typical remote MCP client tool-call timeouts (`~60s`). On timeout, return a clear error containing `<ZOS_BASE_URL>/admin/approvals/<id>` so the assistant shows the clickable link to the user to approve and retry the exact same call.
  3. **Atomic single-use retry:** Within the 10-minute TTL (`ZOS_APPROVAL_TTL_S=600`), retrying the exact `(principal_id, client_id, tool, args_digest)` atomically transitions one `approved` row to `consumed` (`FOR UPDATE SKIP LOCKED`) and executes; any second retry is re-gated.
  4. **Out of scope (V1):** Push notifications to the phone are explicitly deferred; the clickable approval link returned in the tool error is the notification channel for V1.
- **Why:** Phone Claude and ChatGPT do not implement MCP elicitation, which previously caused all `require_approval` calls (e.g., `mac_shell_run`, `gmail_send`) to fail. Plugging async approvals directly behind `require_approval` preserves a single policy engine and audit model while keeping owner review mobile-friendly behind the Access-gated `/admin` path.
- **Impact:** Non-elicitation clients can now execute approval-gated tools by approving on `/admin/approvals/<id>` (either while the call is held or via link + retry); elicitation-capable clients are unaffected.

## 2026-11-25 — Dedicated Python interpreter for Mac actuator (V1 tidy)

- **Decision:** Copy uv's CPython 3.12 installation directory into `~/.zos-actuator/python/` and build `~/.zos-actuator/venv` against `~/.zos-actuator/python/bin/python3.12` in `deploy/launchd/install.sh`. Preserve `~/.zos-actuator/python/` across normal re-installs unless `--upgrade-python` is passed.
- **Why:** macOS Full Disk Access (required for reading `~/Library/Messages/chat.db` and AddressBook SQLite databases) is granted per resolved executable binary. Pointing the actuator venv at uv's shared Python 3.12 binary would grant Full Disk Access to every unrelated uv project using that Python version. Because uv's `python3.12` links `@executable_path/../lib/libpython3.12.dylib`, copying the whole interpreter directory isolates the TCC/FDA grant strictly to `zos-actuator`.
- **Impact:** Full Disk Access is granted only to `~/.zos-actuator/python/bin/python3.12`, and upgrading `zos-actuator` code no longer invalidates the FDA grant.

## 2026-11-25 — Client soft-disable (`clients.disabled_at`) to preserve audit and memory authorship (V1 tidy)

- **Decision:** Add `clients.disabled_at TIMESTAMPTZ` (`0006_v1_client_disabled_at.sql`), update `clients_principal_name_bearer_uidx` to apply only while `disabled_at IS NULL`, and expose `zos client disable <client>` + `/admin/clients/disable` (`AdminService.disable_client`). Disabling revokes all active tokens, blocks future auth, hides the client from active lists/policy matrix, and audits `client.disabled` while keeping the `clients` row intact.
- **Why:** Both `events.client_id` and `memories.created_by_client_id` use `ON DELETE SET NULL` (and `pending_approvals.client_id` uses `ON DELETE CASCADE`), and `delete_oauth_client` intentionally rejects bearer clients. Hard-deleting a client therefore wipes client attribution in `/admin/activity` and strips `author` from memories created by that client. Soft-disabling retires test or old clients cleanly while guaranteeing audit attribution and memory authorship survive indefinitely.
- **Impact:** Retired bearer clients (`codex-smoke`, `opus-live-smoke`, `opus-m8-check`) can be disabled cleanly without losing historical audit attribution, while `memory-import` stays active.

## 2026-11-25 — Removed unused `deploy/fly.toml` and throwaway M0 spikes (V1 tidy)

- **Decision:** Delete `deploy/fly.toml`, `deploy/launchd/.gitkeep`, and the throwaway M0 `spikes/` directory (`spikes/hosting/`, `spikes/google/`).
- **Why:** `deploy/fly.toml` was never deployed or verified on Fly.io (V1 self-hosts on Docker Compose + Cloudflare Named Tunnel) and is broken as written: `[build] dockerfile = "Dockerfile"` fails from `deploy/` context because `deploy/Dockerfile` copies `pyproject.toml`, `migrations/`, `catalog/`, and `src/` from the `zos/` parent directory, and `256mb` VM memory is tight alongside Argon2id verification. Both M0 spikes (`spikes/hosting/` and `spikes/google/`) have been superseded by the shipped ASGI server, Google OAuth flow, and integration test suite.
- **Impact:** The repository contains only live, tested code and working deployment templates (`deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/cloudflared/`, `deploy/launchd/`, `deploy/backup/`).










