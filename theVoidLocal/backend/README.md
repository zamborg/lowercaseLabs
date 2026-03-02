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
make fly-selfhosted-pg-bootstrap DB_APP=thevoid-postgres REGION=sjc DB_VOLUME=thevoid_db_data DB_VOLUME_SIZE_GB=20
make fly-pg-sync SOURCE_DATABASE_URL=... TARGET_DATABASE_URL=...
make fly-pg-parity SOURCE_DATABASE_URL=... TARGET_DATABASE_URL=... PARITY_EXTRA_ARGS='--counts-only'
make fly-pg-cutover APP=thevoid DB_APP=thevoid-postgres POSTGRES_DB=thevoid POSTGRES_USER=thevoid POSTGRES_PASSWORD=... DEPLOY_AFTER_SWITCH=1
make fly-mpg-create APP=thevoid DB_APP=thevoid-mpg REGION=sjc
make fly-mpg-attach APP=thevoid DB_APP=thevoid-mpg
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
For Fly Postgres migration helpers, `make fly-pg-sync`, `make fly-pg-parity`, `make fly-pg-cutover`, and `make fly-selfhosted-pg-bootstrap` auto-source `backend/.env` when present.
Set `ADMIN_DATABASE_URL` if you want `/admin` routes to read/write a different Postgres than the main API `DATABASE_URL`.

## Admin viewer
- URL: `http://127.0.0.1:8080/admin`
- Auth: HTTP Basic (`ADMIN_USERNAME` / `ADMIN_PASSWORD`)
- Social dots table: `http://127.0.0.1:8080/admin/dots`
- User lifecycle table: `http://127.0.0.1:8080/admin/users`

## Fly Deployment (Self-hosted Postgres + Persistent Audio V1)
This repo is configured for a single Fly machine with:
- persistent volume mounted at `/data`
- audio object storage at `/data/storage`
- inline worker thread in the API process (`INLINE_WORKER_ENABLED=true`)
- Alembic release migration step (`alembic upgrade head`)
- database provided by `DATABASE_URL` secret (managed or self-hosted)

This avoids cross-machine file visibility issues for local object storage in V1.

### 1) Bootstrap app + volume
```bash
cd backend
./scripts/fly_bootstrap.sh thevoid sjc thevoid_data 10
```

### 2) Bootstrap a self-hosted Postgres app (container image)
```bash
cd backend
POSTGRES_PASSWORD='<db-password>' \
./scripts/fly_selfhosted_postgres_bootstrap.sh thevoid-postgres sjc thevoid_db_data 20 thevoid thevoid
```

### 3) Set backend app secrets
```bash
fly secrets set -a thevoid \\
  JWT_SECRET='<strong-random-secret>' \\
  APPLE_ALLOWED_AUDIENCES='com.lowercaseLabs.theVoid' \\
  OPENAI_API_KEY='<openai-key>' \\
  ADMIN_USERNAME='<admin-user>' \\
  ADMIN_PASSWORD='<admin-pass>'
```

### 4) Run migration service against the target Postgres
```bash
cd backend
DATABASE_URL='postgresql+psycopg://thevoid:<db-password>@thevoid-postgres.internal:5432/thevoid' \
./scripts/migrate.sh
```

### 5) Sync managed Postgres data into self-hosted Postgres
```bash
cd backend
SOURCE_DATABASE_URL='<managed-postgres-url>' \
TARGET_DATABASE_URL='postgresql://thevoid:<db-password>@thevoid-postgres.internal:5432/thevoid' \
./scripts/postgres_sync_managed_to_container.sh
```

### 6) Verify parity before cutover
```bash
cd backend
poetry run python ./scripts/postgres_parity_check.py \
  --source-url '<managed-postgres-url>' \
  --target-url 'postgresql://thevoid:<db-password>@thevoid-postgres.internal:5432/thevoid'
```

### 7) Cut over backend app DATABASE_URL to self-hosted Postgres and deploy
```bash
cd backend
POSTGRES_PASSWORD='<db-password>' \
DEPLOY_AFTER_SWITCH=1 \
./scripts/fly_switch_database_url.sh thevoid thevoid-postgres thevoid thevoid
```

### 8) Verify
```bash
fly status -a thevoid
fly logs -a thevoid
curl https://thevoid.fly.dev/health
```

### Model hosting (for iOS on-device model download)
- Place model files on the Fly volume at `/data/model_assets`.
- Public listing endpoint: `GET /models`
- Public file endpoint: `GET /models/{filename}`

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
- `postgres_sync_managed_to_container.sh` is a snapshot sync. For strict cutover, run it during a write freeze window.
