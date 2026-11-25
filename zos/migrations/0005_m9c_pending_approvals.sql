-- M9c: Async tool-call approvals for MCP clients without elicitation support.
-- Arguments are encrypted at rest with the CredentialVault AEAD (AES-256-GCM)
-- and purged (set to NULL) once the approval is decided, consumed, or expired.

CREATE TABLE pending_approvals (
    id TEXT PRIMARY KEY,
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    space_id UUID NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    tool TEXT NOT NULL,
    args_digest BYTEA NOT NULL,
    args_ciphertext BYTEA,
    args_nonce BYTEA,
    key_version INT NOT NULL DEFAULT 1,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'denied', 'consumed', 'expired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    decided_at TIMESTAMPTZ,
    consumed_at TIMESTAMPTZ
);

CREATE INDEX pending_approvals_lookup_idx
    ON pending_approvals (principal_id, client_id, tool, args_digest, status, expires_at);

CREATE INDEX pending_approvals_recent_idx
    ON pending_approvals (principal_id, created_at DESC);
