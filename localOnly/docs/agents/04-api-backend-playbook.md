# 04: API Backend Playbook

This process applies to API-first applications (mobile clients, bots, services).

## 1. API Shape

- Expose a simple health endpoint (`/healthz`).
- Keep routes versioned or clearly namespaced if app is non-trivial.
- Return JSON for API endpoints.

## 2. Runtime Expectations

- One process, one port.
- Configuration driven by environment variables.
- No interactive boot sequence.

## 3. Persistence

If persistence is required:

- Use mounted storage path (`/data`) when platform app spec requires volume.
- Make startup resilient when storage is empty.

## 4. Authentication Interaction

In Proxy mode, gateway authenticates before forwarding.

Still recommended for app-level correctness:

- Validate critical auth context where needed.
- Avoid trusting unauthenticated public ingress assumptions.

## 5. Submission Readiness

Before submitting, verify:

- Docker image builds in clean environment.
- `fly.toml` includes `networking_mode` and `internal_port`.
- App starts and responds on declared port.

## 6. Example Structure

```text
echo-pod/
  Dockerfile
  fly.toml
  app/
    main.py
```

Reference implementation:
- `test-apps/echo-pod`
