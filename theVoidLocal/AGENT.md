# theVoid Agent Playbook

This document defines how we collaborate in this repo, how we decide what is "done", and how we operate local + production safely.

## 1) Collaboration Contract

### Default working style
- Move fast, but always leave the repo in a verifiable state.
- Prefer implementation over long planning unless blocked.
- Every meaningful change should include a validation step.
- For production-impacting changes, include deploy verification.
- Default assumption: deploy backend changes to Fly at the end of the task unless the user explicitly asks to skip deploy.

### Communication pattern
- Be explicit about what is being changed, where, and why.
- Surface blockers early with concrete options.
- Report command outcomes in plain language (not just "ran command").

### Definition of done (per task)
- Code change is applied.
- Tests/build checks pass (or gaps are explicitly called out).
- If relevant, migration is applied and deploy is healthy.
- User gets a concise summary + exact next test steps.

## 2) Repo Topology

- `theVoid/` iOS app (SwiftUI, onboarding, recording, social, settings)
- `backend/` FastAPI API + job worker + admin dashboard
- `docker-compose.yml` local container stack (db + migrate + api + worker)
- `fly.toml` production app config for Fly.io
- Root `Makefile` proxies to `backend/Makefile`

## 3) Current Production Status (as of February 18, 2026)

### Fly app
- App: `thevoid`
- URL: `https://thevoid.fly.dev`
- Region: `sjc`
- Health endpoint: `GET /health`

### Last verified state
- Health response: `{"status":"ok", ...}`
- Machine state: `started`, checks `1/1` passing
- Image: `thevoid:deployment-01KHSE672YXANQD9H8G7FBB3BP`
- Machine version: `13`

### Currently live capabilities
- Apple Sign-In verification with Apple JWKS.
- Entry upload/transcription/insights pipeline.
- Admin dashboard with transcript views.
- Admin account lifecycle controls:
  - Decommission account
  - Recommission account
  - Enforcement: decommissioned users receive `403`.

## 4) Git State + Deployment Reality

Important distinction:
- "Deployed" and "pushed to GitHub main" are separate.
- Production can be healthy even if local commits are not pushed yet.

Current known repo state pattern:
- Local branch may be ahead of `origin/main`.
- GitHub push protection can block pushes if old history contains secrets.

Rule:
- Always confirm both when needed:
  - Deploy state (`make fly-status`, `make fly-health`)
  - Git sync state (`git status --branch`, `git push`)

## 5) Daily Working Loop

### Local backend
```bash
make deps
make stack
make test
```

### Local docker stack (preferred for parity)
```bash
make compose-up
make compose-logs
make compose-down
```

### iOS loop
- Run in Simulator for fast UI iteration.
- Run on physical device for Sign in with Apple and end-to-end checks.

### Pre-deploy minimum checks
```bash
cd backend
source .venv/bin/activate
pytest -q
```

## 6) Deployment Playbook (Fly)

### Deploy
```bash
make fly-deploy APP=thevoid
```

### Verify
```bash
make fly-status APP=thevoid
make fly-health APP=thevoid
make fly-logs APP=thevoid
```

### Required production env assumptions
- `ENVIRONMENT=production`
- `ALLOW_DEV_IDENTITY_TOKENS=false`
- `AUTO_CREATE_SCHEMA=false` (Alembic is source of truth)
- Valid `DATABASE_URL` (Postgres, not sqlite)

### Key secrets
- `JWT_SECRET`
- `APPLE_ALLOWED_AUDIENCES`
- `OPENAI_API_KEY`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`

## 7) Auth Model

### Local testing mode
- Dev identity tokens (`dev-*`) accepted only outside production.

### Production mode
- Apple identity token must validate:
  - signature
  - `iss`
  - `aud`
  - token timing claims
  - nonce (when provided)

### Audience handling
- `APPLE_ALLOWED_AUDIENCES` supports comma-separated values.
- Backend validates token against each audience value.

## 8) Admin Dashboard Operations

### URLs
- Overview: `/admin`
- Transcripts: `/admin/transcripts`
- Users + lifecycle controls: `/admin/users`

### Lifecycle controls
- Decommission:
  - Adds user to `account_decommissions`.
  - Optional reason can be recorded.
  - User is blocked from API and future Apple auth sessions (`403`).
- Recommission:
  - Removes user from `account_decommissions`.
  - User can sign in/use API again.

### Operational use case
- Safe internal testing loops across devices without deleting data.
- Temporarily quarantine noisy test accounts while preserving history.

## 9) Schema + Migration Discipline

Rules:
- Any persistent model change must ship with an Alembic migration.
- Never rely on SQLAlchemy `create_all` in production.
- Release command applies migrations on deploy.

Migration commands:
```bash
make migration-new MSG="describe schema change"
make migrate
```

## 10) iOS Testing Notes (Prod)

### Required
- Paid Apple Developer team selected in Xcode (not Personal Team).
- App ID has Sign in with Apple enabled.
- Bundle ID matches backend audience expectations.

### Fast smoke test
1. Backend toggle in app set to `Prod`.
2. Sign in with Apple succeeds.
3. Submit reflection.
4. Transcript appears in app.
5. Transcript visible in `/admin/transcripts`.
6. Invite/social flow across two devices/simulators.

## 11) Incident Checklist

### Symptom: `Invalid Apple identity token`
- Check `APPLE_ALLOWED_AUDIENCES`.
- Confirm app bundle identifier.
- Confirm paid team + SIWA capability + provisioning profile.
- Verify prod does not accept `dev-*` tokens (expected).

### Symptom: invite failures
- Check UTC handling, token expiry, and logs for `/friends/accept`.

### Symptom: deploy healthy but behavior stale
- Confirm deployed image in `fly status`.
- Confirm migration applied.
- Confirm you are hitting prod URL from app toggle.

## 12) Security + Hygiene Rules

- Never commit secrets.
- Rotate secrets immediately if accidental overwrite/leak risk occurs.
- Keep admin credentials private; use Basic auth only over HTTPS.
- Avoid destructive git operations on unknown local changes.

## 13) What to Update in This Doc

Update this file when any of the following changes:
- Deploy topology
- Auth behavior
- Admin capabilities
- Required operational commands
- "Current Production Status" section

This file is intended to be the first place to read before any new implementation session.
