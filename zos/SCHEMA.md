# ZOS — V1 Postgres Schema

Postgres 17 (self-hosted via Docker Compose; compatible with any Postgres 16+). `uuid` PKs via `gen_random_uuid()`, `timestamptz NOT NULL DEFAULT now()`. `psycopg` 3, plain numbered SQL migrations (`migrations/0001_init.sql` through `0006_v1_client_disabled_at.sql`), no ORM. Extension: `pg_trgm`.

Domain rows carry `(principal_id, space_id)`; auth rows carry `principal_id`.

## Identity and auth

- **`principals`** — `id, slug UNIQUE, display_name, created_at`
- **`spaces`** — `id, principal_id FK, slug, name, created_at` · UNIQUE `(principal_id, slug)`. V1: one default space per principal.
- **`clients`** — `id, principal_id FK, default_space_id FK → spaces, name, kind ('oauth'|'bearer'), oauth_client_id TEXT UNIQUE NULL, redirect_uris TEXT[] NOT NULL DEFAULT '{}', trust_ceiling ('owner'|'untrusted') DEFAULT 'owner', created_at, last_seen_at, disabled_at NULL` · UNIQUE `(principal_id, name) WHERE kind = 'bearer' AND disabled_at IS NULL` (soft-disabling via `disabled_at` retires a client while preserving `events.client_id` and `memories.created_by_client_id`).
- **`oauth_codes`** — `code_hash BYTEA PK, client_id FK, code_challenge, redirect_uri, scopes TEXT[], resource, expires_at, used_at NULL` (short-lived PKCE codes).
- **`tokens`** — `id, principal_id FK, client_id FK NULL, provider_id FK NULL, kind ('access'|'refresh'|'bearer'|'actuator'), token_hash BYTEA UNIQUE, family_id UUID NULL, scopes TEXT[], resource, expires_at NULL, rotated_at NULL, revoked_at NULL, last_used_at, created_at` · CHECK exactly one of `client_id` / `provider_id` · INDEX `(family_id)`. Refresh rotation marks used tokens with `rotated_at`; re-presenting a rotated token revokes the entire family. Actuator tokens bind to `provider_id`.

## Providers and credentials

- **`credentials`** — `id, principal_id FK, issuer TEXT CHECK (length(trim(issuer)) > 0), account TEXT, scopes TEXT[], ciphertext BYTEA, nonce BYTEA, key_version INT, access_expires_at, updated_at` · UNIQUE `(principal_id, issuer, account)`. Stores AES-256-GCM encrypted upstream secrets (`google`, `mcp_bearer`, `mcp_oauth`). Separate from `providers` so one Google sign-in serves both `gcal` and `gmail`.
- **`providers`** — `id, principal_id FK, space_id FK, slug UNIQUE per principal (= tool prefix), kind ('hosted'|'upstream'|'actuator'), credential_id FK NULL, config JSONB DEFAULT '{}', enabled BOOL DEFAULT true, last_seen_at, created_at`. Remote MCP upstreams store `{runtime: "remote", transport: "mcp", url, auth_mode, default_tier, ...}` in `config` (zero secrets).

## Policy and approvals

- **`tool_tiers`** — `principal_id FK, tool_pattern TEXT, tier ('read'|'write'|'external')` · PK `(principal_id, tool_pattern)`. Global fact about tools; longest literal match wins; no match → `external`. Seeded on init and extended via `0002` (`gcal_freebusy`) and `0003` (`memory_list`), plus manifest patterns on catalog install.
- **`client_policies`** — `principal_id FK, client_id FK NULL (NULL = all clients), tool_pattern TEXT, decision ('allow'|'deny'|'require_approval')` · UNIQUE NULLS NOT DISTINCT `(principal_id, client_id, tool_pattern)`. Resolution: client-specific match → all-clients match → tier default (`read`/`write` → `allow`, `external` → `require_approval`). `trust_ceiling = 'untrusted'` downgrades `require_approval` → `deny`.
- **`pending_approvals`** — `id TEXT PK (unguessable), principal_id FK, space_id FK, client_id FK, tool TEXT, args_digest BYTEA, args_ciphertext BYTEA NULL, args_nonce BYTEA NULL, key_version INT, status ('pending'|'approved'|'denied'|'consumed'|'expired'), created_at, expires_at, decided_at NULL, consumed_at NULL`. Stores AEAD-encrypted tool arguments while pending; `args_ciphertext` and `args_nonce` are immediately purged (`NULL`) on decision, consumption, or expiration.

## Stores and audit

- **`memories`** — `id, principal_id FK, space_id FK, key TEXT, content TEXT, tags TEXT[] DEFAULT '{}', fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', key || ' ' || content)) STORED, created_by_client_id FK NULL ON DELETE SET NULL, created_at, updated_at` · UNIQUE `(space_id, key)` · GIN(`fts`), GIN(`content gin_trgm_ops`), GIN(`tags`).
- **`events`** — `id BIGINT GENERATED ALWAYS AS IDENTITY PK, principal_id, space_id, client_id NULL ON DELETE SET NULL, provider_id NULL ON DELETE SET NULL, type TEXT, tool TEXT NULL, decision TEXT NULL, outcome ('ok'|'error'|'denied'), args_digest BYTEA NULL, payload JSONB DEFAULT '{}', created_at` · INDEX `(principal_id, id)`. Append-only (no `UPDATE`/`DELETE` in app code). Raw tool arguments are never stored in `events`.

Event types include: `tool.called`, `client.registered`, `client.authorized`, `client.disabled`, `client.deleted`, `token.issued`, `token.revoked`, `auth.failed`, `provider.connected`, `provider.disconnected`, `provider.backend_changed`, `credential.connected`, `credential.refreshed`, `policy.changed`, `tier.changed`, `elicitation.passthrough`, `memory.imported`, `backup.completed`, `upstream.added`, `upstream.token_set`, `upstream.oauth_client_set`, `upstream.disconnected`, `upstream.removed`, `catalog.installed`, `approval.requested`, `approval.decided`.
