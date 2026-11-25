# ZOS Connector Catalog

Checked-in manifests for official remote MCP servers live in `catalog/connectors/*.toml`. Nothing in the catalog is installed by default — install only the connectors you want from **`/admin/upstreams`** ("Available Connectors") or with the CLI:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos catalog list
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos catalog install linear
# Install a second instance under a different tool prefix:
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos catalog install linear --as worklinear
```

## How to add a connector

1. **Verify the endpoint first.** Confirm the server speaks Streamable-HTTP MCP over HTTPS and check its auth behavior (public `none`, `bearer` PAT, OAuth with Dynamic Client Registration, or OAuth requiring pre-registered client credentials).
2. **Create `catalog/connectors/<slug>.toml`** following the manifest format below.
3. **Validate with the test suite:** `uv run pytest tests/integration/test_m9b_catalog.py` validates every checked-in `.toml` file against the Pydantic schema in `src/zos/catalog.py`.

## Manifest format (`manifest_version = 1`)

### Example: OAuth with Dynamic Client Registration

```toml
manifest_version = 1
slug = "linear"
name = "Linear"
description = "Official Linear remote MCP server for issues, projects, cycles, teams, and comments."
docs_url = "https://linear.app/docs/mcp"

[runtime]
kind = "remote"
url = "https://mcp.linear.app/mcp"

[auth]
mode = "oauth"

[tiers]
default = "external"
"*_list*" = "read"
"*_get*" = "read"
"*_search*" = "read"
```

### Example: Bearer token (fine-grained PAT)

```toml
manifest_version = 1
slug = "github"
name = "GitHub"
description = "Official GitHub remote MCP server for repositories, issues, and pull requests."
docs_url = "https://github.com/github/github-mcp-server"

[runtime]
kind = "remote"
url = "https://api.githubcopilot.com/mcp/"

[auth]
mode = "bearer"
token_help = "Create a fine-grained Personal Access Token with minimum required scopes."

[tiers]
default = "external"
"*_list*" = "read"
"*_get*" = "read"
"*_search*" = "read"
```

### Example: OAuth without DCR (pre-registered client)

For servers whose OAuth authorization server does not expose a `registration_endpoint`, set `requires_client_registration = true` and provide `client_help`. ZOS will show Client ID and Client Secret inputs on `/admin/upstreams` (stored encrypted in the vault) before clicking **Connect**.

```toml
[auth]
mode = "oauth"
requires_client_registration = true
client_help = "Create an OAuth app with redirect URI <ZOS_BASE_URL>/connect/<slug>/callback, then paste Client ID & Secret."
```

## Field reference

| Field | Required | Rules |
|---|---|---|
| `manifest_version` | Yes | Must be `1`. |
| `slug` | Yes | Matches `^[a-z][a-z0-9]{1,15}$` (lowercase letters & digits, no underscores, not `zos` or a built-in provider slug). Must match the filename `<slug>.toml`. |
| `name` | Yes | Human-readable connector name. |
| `description` | Yes | Short one-line summary shown in `/admin/upstreams` and `zos catalog list`. |
| `docs_url` | Yes | `https://` link to the upstream provider's MCP documentation. |
| `runtime.kind` | Yes | Must be `"remote"`. |
| `runtime.url` | Yes | `https://` URL of the remote Streamable-HTTP MCP endpoint. |
| `auth.mode` | Yes | One of `"none"`, `"bearer"`, or `"oauth"`. |
| `auth.requires_client_registration` | No | Boolean (default `false`); valid only when `auth.mode = "oauth"`. |
| `auth.token_help` / `auth.client_help` | No | Short setup instructions shown in `/admin/upstreams`. |
| `tiers.default` | Yes | Default tier for unmatched tools: `"read"`, `"write"`, or `"external"` (prefer `"external"` unless the server is strictly read-only). |
| `tiers.<pattern>` | No | Exact tool names or glob patterns mapped to `"read"`, `"write"`, or `"external"`. Patterns are resolved against exposed `<slug>_<tool>` names on initial connect and on upstream `notifications/tools/list_changed`. |

## Tier safety guarantees

- Manifest patterns **never overwrite** a tool tier you have edited manually on `/admin/policy`.
- Removing an installed connector or module automatically removes its manifest-managed tier rows while keeping any tiers you customized by hand.
- Malformed manifests are logged and skipped without preventing ZOS from starting.

---

# Container Module Catalog (`catalog/modules/*.toml`)

Installable container modules live in `catalog/modules/<slug>.toml` using **`manifest_version = 2`** (`runtime.kind = "container"`). Install them from **`/admin/modules`** or the CLI:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module list --catalog
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module install inbox
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module disable inbox
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module enable inbox
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module uninstall inbox          # keeps Postgres schema
docker compose --env-file .env -f deploy/docker-compose.yml exec zos uv run --no-sync zos module uninstall inbox --purge  # drops schema, role, and secrets
```

## How to write a module

1. **Build a standalone MCP server** (in any language) with zero imports from `zos`:
   - Serve Streamable-HTTP MCP on `MCP_PORT` at `MCP_PATH` (default `/mcp`).
   - Optionally serve a human UI on `UI_PORT` (respecting `X-Forwarded-Prefix: /admin/m/<slug>` for links, redirects, and CSRF cookie paths).
   - **Owner visibility primitive:** every module that stores data **must** ship an owner UI under `/admin/m/<slug>` that exposes **all** of its underlying state (including client-to-client data, not just owner-targeted rows), and owner reads must not mutate any client's per-client state (such as unread flags).
   - Copy the stdlib-only verifier [zos_module_sdk/caller.py](../zos_module_sdk/caller.py) (or port its HMAC-SHA256 verification) to verify `params._meta["zos/caller"]` on `tools/call` and `X-ZOS-Caller` / `X-ZOS-Caller-Signature` on UI requests using `ZOS_MODULE_SECRET`. Refuse to attribute unsigned (system) MCP calls to any client.
   - If `[storage] postgres = true`, connect using `ZOS_MODULE_DATABASE_URL` (whose `search_path` is already scoped to your isolated `mod_<slug>` schema).
2. **Add a manifest at `catalog/modules/<slug>.toml`**:
   - Registry images **must** be digest-pinned (`image = "ghcr.io/org/mod@sha256:<64-hex>"`).
   - First-party in-repo modules may set `build = "modules/<slug>"` (ZOS builds the context locally and records the resulting image ID as the pin).
   - Network egress is **off by default** (`[network] egress = false`). Enable it only when the module must reach the internet.
   - Because module UIs are served same-origin under `/admin/m/<slug>/`, only catalog manifests can be installed.
3. **See `modules/inbox/` and `catalog/modules/inbox.toml`** for a complete working example.

## Manifest v2 example (`catalog/modules/inbox.toml`)

```toml
manifest_version = 2
slug = "inbox"
name = "Inbox"
version = "0.1.0"
description = "Async email-like mailboxes shared between you and your agents."

[runtime]
kind = "container"
build = "modules/inbox"      # or: image = "ghcr.io/example/mod@sha256:<64-hex>"
mcp_port = 8081
mcp_path = "/mcp"
ui_port = 8082               # optional

[storage]
postgres = true              # provisions isolated role + schema mod_<slug>

[network]
egress = false               # default false (internal zos_modules network only)

[ui]
title = "Inbox"
path = "/"

[[config]]                   # optional extra config fields shown at install time
name = "EXAMPLE_OPTION"
secret = false
help = "Optional setting"

[tiers]
default = "write"
"inbox_list" = "read"
"inbox_read" = "read"
"inbox_directory" = "read"
```
