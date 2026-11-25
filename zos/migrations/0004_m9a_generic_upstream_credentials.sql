-- Allow generic AEAD-encrypted upstream credentials (e.g. 'mcp_bearer', 'mcp_oauth', future V2 modules)
-- alongside 'google' in the shared credentials vault table.

ALTER TABLE credentials DROP CONSTRAINT IF EXISTS credentials_issuer_check;
ALTER TABLE credentials ADD CONSTRAINT credentials_issuer_check CHECK (length(trim(issuer)) > 0);
