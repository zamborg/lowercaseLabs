# Sovereign App Store MVP

Platform for provisioning isolated app pods on Fly.io with a Control Plane, Data Plane edge router, and client harnesses.

## Current State

Implemented:

1. Backend foundation with uv + FastAPI + PostgreSQL models + JWT auth
2. Fly Machines integration for create/destroy + optional volume/public route setup
3. Data Plane replay router (`fly-replay`) for Proxy mode
4. iOS Expo harness with dynamic route-aware SDK
5. Web support for browser clients (CORS + cookie session + web harness)
6. Developer repo ingestion pipeline (`parse` + `from-chart`) with local and remote (git clone) support
7. Local infrastructure and migrations (Docker Compose + Alembic)
8. Agent playbook docs for generating compatible applications

## Repository Layout

- `backend/`: Control Plane + Data Plane services
- `ios-app/`: iOS client harness (Expo/TypeScript)
- `web-harness/`: browser harness for end-to-end testing without iOS
- `test-apps/`: sample app charts (`echo-pod`, `web-pod`)
- `docs/agents/`: coding-agent implementation playbook and templates
- `docker-compose.yml`: local Postgres/Redis stack

## Quick Start

### 1) Local dependencies + migrations

```bash
make dev-up
```

### 2) Run backend services

```bash
cd backend
cp .env.example .env
uv run control-plane
# terminal 2
uv run data-plane
```

### 3) Run browser harness

```bash
cd web-harness
python3 -m http.server 4173
```

Open `http://localhost:4173`.

### 4) (Optional) Run iOS harness

```bash
cd ios-app
cp .env.example .env
npm install
npm run ios
```

## Key Endpoints

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/web-session` (sets cookie for browser routing)
- `DELETE /auth/web-session`
- `POST /apps/charts/parse`
- `POST /apps/from-chart`
- `POST /pods/provision`
- `GET /pods/routing/apps/{app_slug}`
- Data Plane: `MATCH /{app_slug}/{path:path}`

## Developer Ingestion Paths

### Parse chart metadata only

`POST /apps/charts/parse`

- local source: `repo_path`
- remote source: `repo_url` + optional `ref`

### Parse + register app

`POST /apps/from-chart`

Creates an app record using parsed metadata plus explicit overrides.

## Browser/Web App Support

- Data Plane accepts bearer token **or** gateway session cookie.
- Use `POST /auth/web-session` to set browser cookie.
- Browser apps should use `credentials: include` for cookie mode.

## Agent Documentation

If you are using coding agents to build applications for this platform, start here:

- [docs/agents/README.md](/Users/zubinaysola/Documents/personal/lowercaseLabs/localOnly/docs/agents/README.md)

## Demo App Path

- API app: `test-apps/echo-pod`
- Web app: `test-apps/web-pod`

Both are compatible with the chart parser and registration flow.

Demo automation helpers:
- `cd backend && uv run manual-e2e`
- `cd backend && uv run manual-web-e2e`
