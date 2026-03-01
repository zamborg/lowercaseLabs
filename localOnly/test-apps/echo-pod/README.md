# Echo Pod (Developer Test App)

Minimal single-tenant app chart for Sovereign App Store manual E2E testing.

## Files
- `Dockerfile`
- `fly.toml` (`[metadata].networking_mode = "proxy"`)
- `app/main.py` FastAPI app listening on `8080`

## Deploy Image

Build and push to any image registry reachable from Fly Machines, then register that image via Control Plane `POST /apps`.
