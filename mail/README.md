# Agentic Mail

Minimal agent-assisted email workflow for fast triage and delegated replies.

## One-command local run

- `docker compose up --build`
- `uv run python -m services.tui` (or `docker compose run tui`)

## Setup

1. Copy `.env.example` to `.env` and set values for your IMAP/SMTP credentials if you want real sync.
2. Install Python deps with uv:
   - `uv sync`
3. Start services:
   - `docker compose up --build`
4. Ingest emails:
   - Fixtures: `uv run python services/mail-sync/main.py --source fixtures`
   - IMAP: `uv run python services/mail-sync/main.py --source imap`
   - Docker: `docker compose run mail-sync`
5. Launch the TUI:
   - `uv run python -m services.tui`
   - or `docker compose run tui`

## Real email sync (IMAP)

Set these in `.env`:

```
IMAP_HOST=imap.gmail.com
IMAP_USER=you@gmail.com
IMAP_PASSWORD=your-app-password
IMAP_PORT=993
IMAP_MAILBOX=INBOX
IMAP_SINCE_DAYS=30
IMAP_LIMIT=200
```

Notes for Gmail:

- Enable IMAP in Gmail settings.
- Use an app password (required for accounts with 2FA).

## Demo script

1. Run TUI, use `j/k` to move through emails.
2. Hit `g` to delegate a reply, then `y` to send.
3. Hit `r` to remember an email and confirm it appears in notes (`n`).
4. Use `a` to archive, `d` to delete, `s` to snooze.

## Notes

- API auth uses `Authorization: Bearer <DEV_TOKEN>` from `.env`.
- Fixtures live in `services/mail-sync/fixtures/`.
- Agent jobs are queued in Redis and processed by `services/agent`.
