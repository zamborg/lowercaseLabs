# Sovereign App Store Backend

FastAPI implementation of:

- Control Plane (auth, app registry, chart ingestion, pod lifecycle)
- Data Plane (JWT/cookie validation + `fly-replay` routing)

## Stack

- Python 3.12+
- FastAPI
- PostgreSQL via async SQLAlchemy
- Redis (optional route cache)
- Fly.io Machines API
- Alembic migrations
- uv dependency management

## Local Bring-Up

From repository root:

```bash
make dev-up
```

Or manually:

```bash
docker compose up -d postgres redis
cd backend
uv sync
uv run db-upgrade
```

## Run Services

```bash
uv run control-plane
# terminal 2
uv run data-plane
```

Control Plane docs: `http://localhost:8000/docs`
Data Plane docs: `http://localhost:8001/docs`

## Core Endpoints

### Authentication

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /auth/web-session` (set browser cookie)
- `DELETE /auth/web-session`

### App Registry

- `GET /apps`
- `POST /apps`
- `POST /apps/charts/parse`
- `POST /apps/from-chart`

### Pods

- `POST /pods/provision`
- `GET /pods`
- `DELETE /pods/{pod_id}`
- `POST /pods/{pod_id}/members`
- `GET /pods/routing/apps/{app_slug}`

### Data Plane

- `MATCH /{app_slug}/{path:path}`
- Returns `202` with header: `fly-replay: instance=<machine_id>`

## Chart Ingestion

`/apps/charts/parse` supports:

- local parse by `repo_path`
- remote parse by `repo_url` + optional `ref` (temporary git clone)

`/apps/from-chart` supports parse + app registration in one request.

## Browser Support

Data Plane accepts either:

- `Authorization: Bearer <jwt>`
- web session cookie (set via `POST /auth/web-session`)

For browser clients, set `credentials: include`.

## Config

See `.env.example` for all variables, including:

- CORS settings (`CORS_ALLOWED_ORIGINS`)
- cookie settings (`GATEWAY_SESSION_COOKIE_*`)
- repo parse controls (`REPO_CLONE_*`, `REPO_ALLOWED_HOSTS`)

## Scripts

- `uv run db-upgrade`
- `uv run init-db` (legacy direct metadata create)
- `uv run manual-e2e`
- `uv run manual-web-e2e`
