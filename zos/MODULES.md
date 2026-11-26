# ZOS Module Contract v1.1

A ZOS module is a standalone MCP server (in any language) packaged as a container with a checked-in manifest. ZOS owns authentication, policy, approvals, audit logging, Postgres schema provisioning, and reverse-proxying the module's human UI under `/admin/m/<slug>/`.

Core (`src/zos/`) never imports module code, and modules never import `zos`.

---

## 1. Manifest (`manifest_version = 2`)

Module manifests live in `catalog/modules/<slug>.toml` (filename stem must match `slug`).

```toml
manifest_version = 2
slug = "inbox"                     # ^[a-z][a-z0-9]{1,15}$, not 'zos' or a built-in slug
name = "Inbox"
version = "0.1.0"
description = "Async internal mailboxes shared between agents and the owner."
docs_url = ""                      # optional https:// URL

[runtime]
kind = "container"
# Set EXACTLY ONE of `image` (production/hosted) or `build` (first-party local dev):
image = "ghcr.io/zamborg/zos-inbox@sha256:<64-hex>"
# build = "modules/inbox"          # relative path under modules/<slug>
mcp_port = 8081                    # required: Streamable-HTTP MCP port
mcp_path = "/mcp"                  # optional, default "/mcp"
ui_port = 8082                     # optional: human UI HTTP port

[storage]
postgres = true                    # default false; provisions isolated role & schema mod_<slug>
extensions = ["pg_trgm"]           # optional (Contract v1.1); allowlisted trusted extensions in shared `extensions` schema

[network]
egress = false                     # default false; true attaches zos_modules_egress

[ui]
title = "Inbox"                    # label shown in /admin nav and Overview
path = "/"                         # optional, default "/"

[[config]]                         # optional install-time config fields
name = "UPPER_SNAKE_NAME"
secret = false                     # stored encrypted in AEAD vault either way
required = false
help = "Short hint shown in /admin/modules"

[tiers]
default = "write"                  # read | write | external (default: external)
"inbox_list" = "read"              # exact exposed <slug>_<tool> names or glob patterns
"inbox_read" = "read"
"inbox_directory" = "read"
"inbox_send" = "write"
"inbox_archive" = "write"
```

### Image pinning & publishing
- Registry images **must** be digest-pinned (`<repo>@sha256:<64-hex>`). Unpinned tags (`:latest`, `:v1`) are rejected at manifest load time.
- First-party in-repo modules may set `build = "modules/<slug>"` for local development; `DockerModuleHost` builds the context and records the resulting image ID as `image_pin`.
- Production/hosted first-party images are published to GHCR via a tag-triggered CI workflow (`ghcr.io/zamborg/zos-<slug>:<tag>`) and pinned by `@sha256:<digest>` in `catalog/modules/<slug>.toml`.

---

## 2. Environment handed to a module

ZOS writes the container spec itself (never taken from the manifest: `Privileged=false`, `CapDrop=["ALL"]`, `no-new-privileges`, zero host binds/mounts, memory cap, `restart=unless-stopped`) and injects:

| Variable | When set | Description |
|---|---|---|
| `ZOS_MODULE_SLUG` | Always | Module slug (e.g. `inbox`). |
| `ZOS_MODULE_SECRET` | Always | Per-module random secret (stored in ZOS's AEAD vault) for verifying caller signatures. Preserved across upgrades. |
| `ZOS_MODULE_DATABASE_URL` | `storage.postgres = true` | Postgres DSN for dedicated role `mod_<slug>` whose `search_path` is locked to schema `mod_<slug>` (plus `extensions` when `storage.extensions` is declared). The role has zero access to `public` core tables (`credentials`, `events`, `memories`, `tokens`). |
| `MCP_PORT`, `MCP_PATH` | Always | Port and path where the module serves Streamable-HTTP MCP. |
| `UI_PORT` | `ui_port` set | Port where the module serves its human UI. |
| `<CONFIG_FIELD>` | Configured `[[config]]` | Install-time configuration values (stored encrypted in the ZOS vault). |

---

## 3. MCP caller identity (`params._meta["zos/caller"]`)

Every ZOS → module `tools/call` request carries an HMAC-SHA256-signed envelope inside MCP `params._meta["zos/caller"]` (transport-independent, spec-native):

```json
{
  "payload": {
    "principal_id": "<uuid>",
    "client_id": "<uuid>",
    "client_name": "<client name>",
    "tool": "<raw local tool name>",
    "args_digest": "<lowercase hex SHA-256 of canonical JSON arguments>",
    "ts": 1700000000
  },
  "sig": "<lowercase hex HMAC-SHA256(ZOS_MODULE_SECRET, canonical_json(payload))>"
}
```

- **Canonical JSON:** `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
- **Verification rules:**
  1. Constant-time HMAC check against `ZOS_MODULE_SECRET`.
  2. Timestamp skew within `±60` seconds.
  3. `payload.tool` equals the requested MCP tool name.
  4. `payload.args_digest` equals SHA-256 of canonical JSON of `params.arguments or {}`.
- **Unsigned (system) calls:** Requests without `_meta["zos/caller"]` (e.g. `tools/list`) are system calls. Modules **must** refuse to attribute them to a client.
- **Reserved names:** Because OAuth DCR clients choose their own display names, modules with reserved owner/shared identities (such as `owner` and `general` in Inbox) must refuse those names over MCP and allow acting as `owner` only over the authenticated HTTP UI proxy.
- **Reference verifier:** Copy the standalone stdlib-only verifier [`zos_module_sdk/caller.py`](zos_module_sdk/caller.py) (zero dependency on `zos`).

---

## 4. UI proxy headers & cookies (`/admin/m/<slug>/...`)

When `ui_port` is declared, ZOS reverse-proxies `/admin/m/<slug>` and `/admin/m/<slug>/{path}` behind the owner `/admin` session:

- **Stripped before forwarding:** `zos_admin_session` (and any `zos_admin*` cookie), `Authorization`, and any client-supplied `X-ZOS-Caller*` or `X-Forwarded-*` headers.
- **Injected by ZOS:**
  - `X-ZOS-Caller`: base64url canonical JSON `{"client_id":"owner","client_name":"owner","principal_id":"<uuid>","ts":<epoch>}`.
  - `X-ZOS-Caller-Signature`: `sha256=<lowercase hex HMAC-SHA256(ZOS_MODULE_SECRET, X-ZOS-Caller)>`.
  - `X-Forwarded-Prefix`: `/admin/m/<slug>`.
  - `X-Forwarded-Proto`: `http` or `https` (from `ZOS_BASE_URL`).
  - `X-Forwarded-Host`: public host[:port] (from `ZOS_BASE_URL`, Contract v1.1) so modules that perform upstream OAuth can construct exact callback URLs (`<proto>://<host><prefix>/callback`).
- **Cross-site OAuth callbacks (`SameSite=Strict`):** Because `zos_admin_session` is `SameSite=Strict`, browsers omit it on cross-site redirects back from external OAuth providers. For cookieless top-level `GET` requests to `/admin/m/*` carrying `Sec-Fetch-Site: cross-site`, ZOS returns a minimal same-origin `<meta http-equiv="refresh">` re-navigation page to the exact same path and query; the browser's immediate same-origin follow-up carries `zos_admin_session` without weakening the cookie.
- **Module obligations:**
  - Verify `X-ZOS-Caller` + `X-ZOS-Caller-Signature` on every UI request (except `/healthz`); return `403` otherwise.
  - Prefix all generated links, form actions, redirects, and CSRF cookie paths with `X-Forwarded-Prefix`.
  - Escape all untrusted content and enforce CSRF on mutating `POST` forms using the module's own cookie (e.g. `inbox_csrf`).
- **Response rewriting by ZOS:** `Location` headers targeting `/` or the module origin are rewritten under `/admin/m/<slug>`. Module `Set-Cookie` headers are scoped to `Path=/admin/m/<slug>`, and any cookie named like `zos_admin*` is dropped.
- **Trust model:** Module UIs are served same-origin with `/admin` (WordPress-plugin model), which is why only catalog-pinned manifests may be installed.

---

## 5. Networking

- **`network.egress = false` (default):** Container attaches only to `zos_modules` (`internal: true`), where it can reach `postgres:5432` and receive calls from `zos`, with **zero** internet egress.
- **`network.egress = true`:** Container additionally attaches to `zos_modules_egress` for outbound internet access.
- Containers never publish host ports or join the host network/PID namespace.

---

## 6. Owner visibility primitive

Every module that stores data **must** ship an owner view under `/admin/m/<slug>` exposing **all** of its underlying state — including client-to-client data, not only items addressed to `owner`. Owner reads in the UI must **not** mutate clients' per-client state (such as unread flags).

---

## 7. Schema migrations & upgrades

- **Module-owned migrations:** Modules own their `mod_<slug>` schema and run **idempotent, versioned** SQL migrations automatically on startup (tracking applied versions in a module migration table, e.g. `inbox_schema_migrations`).
- **Upgrades (`zos module upgrade <slug>` / `/admin/modules/upgrade`):** Rebuilds or pulls the new image, recreates the container, and **preserves** the `mod_<slug>` Postgres schema/data, `ZOS_MODULE_SECRET`, `providers` row (`upstream_id`), and any manual owner tier/policy overrides.

---

## 8. Versioning policy

- **`manifest_version = 2`** covers Module Contract v1 and additive v1.x extensions (`v1.1` adds `[storage].extensions` with a shared `extensions` schema for trusted Postgres extensions such as `pg_trgm`). Breaking changes require `manifest_version = 3`.
- **Caller envelope (`zos/caller`, `X-ZOS-Caller`)** is frozen for Contract v1; any breaking wire change will use a new key/header version while supporting v1 during migration.
- **Module `version`** follows SemVer (`MAJOR.MINOR.PATCH`) and is recorded in the `modules` table and `module.installed` / `module.upgraded` audit events.
