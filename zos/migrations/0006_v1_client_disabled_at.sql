-- V1 tidy: soft-disable clients so audit client attribution (events.client_id)
-- and memory authorship (memories.created_by_client_id) survive client retirement.

ALTER TABLE clients ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ;

DROP INDEX IF EXISTS clients_principal_name_bearer_uidx;
CREATE UNIQUE INDEX clients_principal_name_bearer_uidx
    ON clients (principal_id, name)
    WHERE kind = 'bearer' AND disabled_at IS NULL;
