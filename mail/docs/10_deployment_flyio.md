# Deployment (Fly.io)

## Fly.io approach (MVP)

- 1 Fly app per service is ideal, but simplest is:
  - deploy api + agent + mail-sync as separate Fly apps
  - use managed Postgres (Fly Postgres) + Redis (Upstash or Fly)
- Store secrets in Fly:
  - DB URL, REDIS URL, ENCRYPTION_KEY, DEV_TOKEN
- For MVP, TUI runs locally pointing to deployed API.

## Operational notes

- mail-sync needs outbound IMAP/SMTP connectivity.
- agent service needs outbound internet to model provider.
- rate-limit send.
