CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email_address TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accounts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id),
    provider TEXT NOT NULL DEFAULT 'imap',
    imap_host TEXT,
    imap_user TEXT,
    imap_encrypted_pass TEXT,
    smtp_host TEXT,
    smtp_user TEXT,
    smtp_encrypted_pass TEXT
);

CREATE TABLE IF NOT EXISTS emails (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id uuid NOT NULL REFERENCES accounts(id),
    message_id TEXT NOT NULL,
    thread_key TEXT,
    "from" JSONB NOT NULL,
    "to" JSONB NOT NULL,
    cc JSONB NOT NULL DEFAULT '[]',
    bcc JSONB NOT NULL DEFAULT '[]',
    subject TEXT,
    date TIMESTAMPTZ,
    snippet TEXT,
    body_text TEXT,
    body_html TEXT,
    labels JSONB NOT NULL DEFAULT '{}',
    is_read BOOLEAN NOT NULL DEFAULT false,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS triage_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id),
    email_id uuid NOT NULL REFERENCES emails(id),
    action TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS categories (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT,
    matching_rules JSONB NOT NULL DEFAULT '{}',
    default_policy_id uuid
);

CREATE TABLE IF NOT EXISTS policies (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    autonomy_level INT NOT NULL DEFAULT 0,
    allowed_actions JSONB NOT NULL DEFAULT '[]',
    reply_template TEXT,
    style_profile JSONB NOT NULL DEFAULT '{}',
    escalation_rules JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agent_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id),
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    input_refs JSONB NOT NULL DEFAULT '{}',
    output JSONB NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS notes_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id),
    title TEXT NOT NULL,
    content_markdown TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS notes_items (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    notes_document_id uuid NOT NULL REFERENCES notes_documents(id),
    email_id uuid NOT NULL REFERENCES emails(id),
    summary TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]',
    citations JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_emails_account_id ON emails(account_id);
CREATE INDEX IF NOT EXISTS idx_emails_thread_key ON emails(thread_key);
CREATE INDEX IF NOT EXISTS idx_triage_email_id ON triage_events(email_id);
CREATE INDEX IF NOT EXISTS idx_agent_jobs_user_id ON agent_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_items_doc ON notes_items(notes_document_id);
