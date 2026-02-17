# Security and privacy

Minimum required, because email.

- Encrypt stored credentials at rest (dev can use a static key; prod uses Fly secrets).
- Do not log raw bodies in plain logs.
- Agent prompt redaction: strip secrets/credentials by pattern.
- Audit trail is immutable: triage_events append-only.
- Provide a data export endpoint later; MVP can be dump notes + triage.
