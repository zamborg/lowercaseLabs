-- M10b: installed modules table

CREATE TABLE modules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id UUID NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
    space_id UUID NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    manifest_version INT NOT NULL,
    version TEXT NOT NULL,
    image_pin TEXT NOT NULL,
    host TEXT NOT NULL CHECK (host IN ('docker', 'manual')),
    status TEXT NOT NULL DEFAULT 'installed'
        CHECK (status IN ('installed', 'pending_attach', 'disabled', 'error')),
    mcp_url TEXT NULL,
    ui_url TEXT NULL,
    ui_title TEXT NULL,
    upstream_id UUID NULL REFERENCES providers(id) ON DELETE SET NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    installed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    disabled_at TIMESTAMPTZ NULL,
    UNIQUE (principal_id, slug)
);

CREATE INDEX modules_principal_status_idx ON modules (principal_id, status);
