# theVoid backend (V1)

## Components
- `app.main`: synchronous API (`/auth`, `/entries`, `/friends`, `/social`)
- `app.worker`: async pipeline worker (`transcription -> insights -> social_aggregation`)
- DB-backed queue (`jobs` table)
- local object storage fallback (`backend/storage`)

## Local run

```bash
cd backend
./scripts/dev_api.sh
```

Run worker in another terminal:

```bash
cd backend
./scripts/dev_worker.sh
```

Or run both together:

```bash
cd backend
./scripts/dev_stack.sh
```

`dev_stack.sh` now runs the dockerized stack (Postgres + migrations + API + worker).
By default it streams logs and stops the stack on Ctrl+C.
Use detached mode if needed:
```bash
cd backend
FOLLOW_LOGS=0 ./scripts/dev_stack.sh
```

For physical-device testing on your LAN:

```bash
cd backend
API_PORT=8080 ./scripts/dev_stack.sh
```

If you need physical-device iOS testing, run API on LAN:

```bash
cd backend
./scripts/dev_api.sh 0.0.0.0 8080
```

### Quick ingest script (upload -> transcribe -> tag)
```bash
cd backend
./scripts/ingest_reflection.py \
  --api-base-url http://127.0.0.1:8080 \
  --identity-token dev-zubin \
  --audio /absolute/path/to/reflection.m4a \
  --duration-seconds 120
```
The script prints status updates and final `tags` + `summary`.

## Migrations (Alembic)
```bash
cd backend
./scripts/migrate.sh
```

Create a new migration:
```bash
cd backend
./scripts/migration_new.sh "describe schema change"
```

Recommended production setting:
- `AUTO_CREATE_SCHEMA=false` (use migrations instead of `create_all`)

## Local Docker stack (Postgres + API + Worker)
From repo root:
```bash
make compose-up
```

This runs:
1. Postgres
2. `alembic upgrade head`
3. API + worker containers

Postgres is exposed on host port `55432` by default to avoid collisions with an existing local Postgres (`POSTGRES_PORT` is configurable).
API is exposed on host port `8080` by default (`API_PORT` is configurable).

Useful commands:
```bash
make compose-ps
make compose-logs
make compose-migrate
make compose-down
```

## Makefile shortcuts
From repo root:
```bash
make help
make deps
make stack
make stack-lan PORT=8080
make test
make migrate
make migration-new MSG="add new column"
make ingest AUDIO=/absolute/path/to/reflection.m4a IDENTITY_TOKEN=dev-zubin
make compose-up
make compose-logs
make compose-down
make fly-bootstrap APP=thevoid
make fly-mpg-create APP=thevoid DB_APP=thevoid-db REGION=sjc
make fly-mpg-attach APP=thevoid DB_APP=thevoid-db
make fly-deploy APP=thevoid
make fly-status APP=thevoid
make fly-logs APP=thevoid
make fly-health APP=thevoid
```

Set Fly secrets via make:
```bash
JWT_SECRET='<strong-random-secret>' \
APPLE_ALLOWED_AUDIENCES='com.lowercaseLabs.theVoid' \
OPENAI_API_KEY='<openai-key>' \
ADMIN_USERNAME='<admin-user>' \
ADMIN_PASSWORD='<admin-pass>' \
make fly-secrets APP=thevoid
```

## Environment
Copy `.env.example` to `.env` and customize.

## Admin viewer
- URL: `http://127.0.0.1:8080/admin`
- Auth: HTTP Basic (`ADMIN_USERNAME` / `ADMIN_PASSWORD`)
- Transcript table: `http://127.0.0.1:8080/admin/transcripts`

## Fly Deployment (Postgres + Persistent Audio V1)
This repo is configured for a single Fly machine with:
- persistent volume mounted at `/data`
- audio object storage at `/data/storage`
- inline worker thread in the API process (`INLINE_WORKER_ENABLED=true`)
- Alembic release migration step (`alembic upgrade head`)
- database provided by `DATABASE_URL` secret (Fly Managed Postgres)

This avoids cross-machine file visibility issues for local object storage in V1.

### 1) Bootstrap app + volume
```bash
cd backend
./scripts/fly_bootstrap.sh thevoid sjc thevoid_data 10
```

### 2) Create and attach Fly Managed Postgres (required)
```bash
fly mpg create --name thevoid-db --region sjc
fly mpg attach thevoid-db -a thevoid
```

### 3) Set app secrets
```bash
fly secrets set -a thevoid \\
  JWT_SECRET='<strong-random-secret>' \\
  APPLE_ALLOWED_AUDIENCES='lowercaseLabs.theVoid' \\
  OPENAI_API_KEY='<openai-key>' \\
  ADMIN_USERNAME='<admin-user>' \\
  ADMIN_PASSWORD='<admin-pass>'
```

### 4) Deploy
```bash
cd /path/to/theVoid
backend/scripts/fly_deploy.sh thevoid
```

### 5) Verify
```bash
fly status -a thevoid
fly logs -a thevoid
curl https://thevoid.fly.dev/health
```

## Notes
- Apple identity tokens are verified server-side with Apple's JWKS (`iss`, `aud`, signature, exp/iat, and optional nonce).
- `APPLE_ALLOWED_AUDIENCES` must include your iOS bundle id / service id (comma-separated if multiple).
- Dev token fallback (`dev-*`) is disabled automatically when `ENVIRONMENT=production`.
- Transcription uses OpenAI when `OPENAI_API_KEY` is set (`OPENAI_TRANSCRIPTION_MODEL` configurable). Without a key, it falls back to a local stub transcript for development.
- Insights use deterministic scoring + theming, and mood tags are selected from a hardcoded 50-tag catalog in `app/tagging.py`.
- If `OPENAI_API_KEY` is set, tags are selected by OpenAI (`OPENAI_INSIGHTS_MODEL`) and clamped to 1-4 tags from that catalog.
- Audio is stored outside Postgres by object key as required.
- The production migration path is Alembic (`alembic upgrade head`), not SQLAlchemy `create_all`.
- Production startup rejects sqlite `DATABASE_URL` to avoid accidental non-persistent deployments.
