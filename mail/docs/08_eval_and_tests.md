# Eval and tests

You want concrete tests your coding agent can implement.

## Unit tests

- policy schema validation
- category matching determinism
- audit logging is append-only

## Integration tests (docker-compose)

1. Ingest fixture emails (from services/mail-sync/fixtures/)
2. Run classification job -> confirm suggested categories
3. Delegate reply with fixed instruction -> confirm draft exists and meets constraints
4. Notes update -> notes doc includes citation to email IDs

## Golden tests (agent behavior)

Create a small curated corpus:

- recruiter email
- scheduling email
- FYI newsletter
- angry customer email

For each:

- expected snooze suggestion range (e.g., 1-3 days)
- expected category suggestion
- expected draft characteristics (contains gratitude + clear next step, etc.)
