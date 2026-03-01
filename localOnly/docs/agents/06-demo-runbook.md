# 06: Demo Runbook

Use this runbook to get a full browser-first demo running quickly.

## 1. Bring Up Platform Dependencies

```bash
make dev-up
```

## 2. Start Platform Services

```bash
cd backend
cp .env.example .env
uv run control-plane
# terminal 2
uv run data-plane
```

## 3. Build and Push Demo App Image

Use `test-apps/web-pod` or your own app.

Example (replace registry):

```bash
cd test-apps/web-pod
docker build -t registry.example.com/web-pod:latest .
docker push registry.example.com/web-pod:latest
```

## 4. Run Web Harness

```bash
cd web-harness
python3 -m http.server 4173
```

Open `http://localhost:4173`.

## 5. Register + Route Demo App

In web harness:

1. Register/login.
2. Set web session cookie.
3. Parse chart (local path or repo URL).
4. Register app from chart (`/apps/from-chart`).
5. Provision pod (currently easiest through API docs `POST /pods/provision`).
6. Resolve route.
7. Open web app.

## 6. Troubleshooting

- `401` on Data Plane: token/cookie missing or expired.
- `404` resolve route: no active pod membership for that app.
- `502` provisioning: Fly API token or app/image config issue.
- CORS issues: ensure harness origin exists in `CORS_ALLOWED_ORIGINS`.
