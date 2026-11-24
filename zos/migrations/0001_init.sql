-- ZOS V0 initial schema (matches SCHEMA.md)

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Identity and auth

CREATE TABLE principals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE spaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (principal_id, slug)
);

CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    default_space_id UUID NOT NULL REFERENCES spaces(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('oauth', 'bearer')),
    oauth_client_id TEXT UNIQUE,
    redirect_uris TEXT[] NOT NULL DEFAULT '{}',
    trust_ceiling TEXT NOT NULL DEFAULT 'owner' CHECK (trust_ceiling IN ('owner', 'untrusted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX clients_principal_name_bearer_uidx
    ON clients (principal_id, name)
    WHERE kind = 'bearer';

CREATE TABLE oauth_codes (
    code_hash BYTEA PRIMARY KEY,
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    code_challenge TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    resource TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ
);

-- Providers and credentials

CREATE TABLE credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    issuer TEXT NOT NULL CHECK (issuer IN ('google')),
    account TEXT NOT NULL,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    ciphertext BYTEA NOT NULL,
    nonce BYTEA NOT NULL,
    key_version INT NOT NULL DEFAULT 1,
    access_expires_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (principal_id, issuer, account)
);

CREATE TABLE providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    space_id UUID NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('hosted', 'upstream', 'actuator')),
    credential_id UUID REFERENCES credentials(id) ON DELETE SET NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT true,
    last_seen_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (principal_id, slug)
);

CREATE TABLE tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    provider_id UUID REFERENCES providers(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('access', 'refresh', 'bearer', 'actuator')),
    token_hash BYTEA NOT NULL UNIQUE,
    family_id UUID,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    resource TEXT,
    expires_at TIMESTAMPTZ,
    rotated_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((client_id IS NOT NULL) <> (provider_id IS NOT NULL))
);

CREATE INDEX tokens_family_id_idx ON tokens (family_id);

-- Policy

CREATE TABLE tool_tiers (
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    tool_pattern TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('read', 'write', 'external')),
    PRIMARY KEY (principal_id, tool_pattern)
);

CREATE TABLE client_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    tool_pattern TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('allow', 'deny', 'require_approval')),
    UNIQUE NULLS NOT DISTINCT (principal_id, client_id, tool_pattern)
);

-- Stores and audit

CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    space_id UUID NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', key || ' ' || content)) STORED,
    created_by_client_id UUID REFERENCES clients(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (space_id, key)
);

CREATE INDEX memories_fts_gin_idx ON memories USING GIN (fts);
CREATE INDEX memories_content_trgm_gin_idx ON memories USING GIN (content gin_trgm_ops);
CREATE INDEX memories_tags_gin_idx ON memories USING GIN (tags);

CREATE TABLE events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    space_id UUID NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    client_id UUID REFERENCES clients(id) ON DELETE SET NULL,
    provider_id UUID REFERENCES providers(id) ON DELETE SET NULL,
    type TEXT NOT NULL,
    tool TEXT,
    decision TEXT,
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'error', 'denied')),
    args_digest BYTEA,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX events_principal_id_idx ON events (principal_id, id);
