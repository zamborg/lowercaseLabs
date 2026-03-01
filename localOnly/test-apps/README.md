# Test Applications

This directory contains developer-style app charts you can use for real end-to-end testing.

## Included

- `echo-pod/`: minimal FastAPI single-tenant backend with `Dockerfile` and `fly.toml`
- `web-pod/`: minimal browser-facing web application with same pod model

## Suggested Workflow

1. Build and push the image from `echo-pod` or `web-pod`.
2. Ensure your Control Plane `.env` has valid Fly credentials.
3. Run `uv run manual-e2e` from `backend/` with `E2E_DOCKER_IMAGE` and `E2E_FLY_APP_NAME` set.
4. Use the iOS harness or `web-harness` to authenticate and call the provisioned pod.
