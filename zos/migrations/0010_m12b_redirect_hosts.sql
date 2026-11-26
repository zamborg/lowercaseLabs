-- M12b: move OAuth DCR redirect host allowlist to Postgres and record rejected attempts.

CREATE TABLE IF NOT EXISTS allowed_redirect_hosts (
    host TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dcr_rejected_attempts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    host TEXT NOT NULL,
    client_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (host, client_name)
);

INSERT INTO allowed_redirect_hosts (host, source)
VALUES ('chatgpt.com', 'default'), ('claude.ai', 'default')
ON CONFLICT (host) DO NOTHING;
