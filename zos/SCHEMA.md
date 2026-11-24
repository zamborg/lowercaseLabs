# ZOS — V0 Postgres schema

Status: final (review 06). Postgres 16+ (Neon is 17). `uuid` PKs via
`gen_random_uuid()`, `timestamptz NOT NULL DEFAULT now()`. psycopg 3,
plain numbered SQL migrations, no ORM. Extension: `pg_trgm`.
Domain rows carry `(principal_id, space_id)`; auth rows carry `principal_id`.

## Identity and auth

- **`principals`** — `id, slug UNIQUE, display_name, created_at`
- **`spaces`** — `id, principal_id FK, slug, name, created_at` ·
  UNIQUE `(principal_id, slug)`. V0: exactly one per principal.
- **`clients`** — `id, principal_id FK, default_space_id FK → spaces, name,
  kind ('oauth'|'bearer'), oauth_client_id TEXT UNIQUE NULL, redirect_uris
  TEXT[] NOT NULL DEFAULT '{}', trust_ceiling ('owner'|'untrusted') DEFAULT
  'owner', created_at, last_seen_at` · UNIQUE `(principal_id, name) WHERE
  kind = 'bearer'` (OAuth clients are keyed by `oauth_client_id`; two installs
  may share a display name)
- **`oauth_codes`** — `code_hash BYTEA PK, client_id FK, code_challenge,
  redirect_uri, scopes TEXT[], expires_at, used_at NULL` (short-lived PKCE codes)
- **`tokens`** — `id, principal_id FK, client_id FK NULL, provider_id FK
  NULL, kind ('access'|'refresh'|'bearer'|'actuator'), token_hash BYTEA
  UNIQUE, family_id UUID NULL, scopes TEXT[], expires_at NULL, rotated_at
  NULL, revoked_at NULL, last_used_at` · CHECK exactly one of client_id /
  provider_id · INDEX `(family_id)`. Refresh rotation: the used token gets
  `rotated_at`; presenting a token with `rotated_at` set revokes the whole
  family. Actuator tokens bind to `provider_id`.

## Providers and credentials

- **`credentials`** — `id, principal_id FK, issuer ('google'), account
  TEXT, scopes TEXT[], ciphertext BYTEA, nonce BYTEA, key_version INT,
  access_expires_at, updated_at` · UNIQUE `(principal_id, issuer, account)`.
  Separate from providers so one Google sign-in serves both the `gcal` and
  `gmail` providers.
- **`providers`** — `id, principal_id FK, space_id FK, slug UNIQUE per
  principal (= tool prefix; V0: 'memory'|'gcal'|'gmail'|'mac'), kind ('hosted'|'upstream'|'actuator'),
  credential_id FK NULL, config JSONB DEFAULT '{}', enabled BOOL DEFAULT
  true, last_seen_at, created_at`

## Policy

- **`tool_tiers`** — `principal_id FK, tool_pattern TEXT, tier
  ('read'|'write'|'external')` · PK `(principal_id, tool_pattern)`. Global
  fact about tools; longest literal match wins; no match → `external`.
  Seeded per provider (see BUILD.md).
- **`client_policies`** — `principal_id FK, client_id FK NULL (NULL = all
  clients), tool_pattern TEXT, decision ('allow'|'deny'|'require_approval')`
  · UNIQUE NULLS NOT DISTINCT `(principal_id, client_id, tool_pattern)`.
  Resolution: client-specific match → all-clients match → tier default
  (read/write → allow, external → require_approval). `trust_ceiling =
  'untrusted'` turns any resolved `require_approval` into `deny`.

## Stores and audit

- **`memories`** — `id, principal_id FK, space_id FK, key TEXT, content
  TEXT, tags TEXT[] DEFAULT '{}', fts TSVECTOR GENERATED ALWAYS AS
  (to_tsvector('english', key || ' ' || content)) STORED,
  created_by_client_id FK NULL, created_at, updated_at` · UNIQUE
  `(space_id, key)` · GIN(fts), GIN(content gin_trgm_ops), GIN(tags)
- **`events`** — `id BIGINT GENERATED ALWAYS AS IDENTITY PK, principal_id,
  space_id, client_id NULL, provider_id NULL, type TEXT, tool TEXT NULL,
  decision TEXT NULL, outcome ('ok'|'error'|'denied'), args_digest BYTEA
  NULL, payload JSONB DEFAULT '{}', created_at` · INDEX `(principal_id, id)`.
  Append-only (no UPDATE/DELETE in app code). Raw args never stored by
  default.

Event types: `tool.called`, `client.registered`, `client.authorized`,
`token.issued`, `token.revoked`, `provider.connected`,
`provider.disconnected`, `credential.connected`, `credential.refreshed`,
`policy.changed`.
