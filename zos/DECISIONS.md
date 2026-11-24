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




