# 05: Delivery Checklist

Coding agents must complete this checklist before handing off an app repository.

## A. Repository Layout

- [ ] `Dockerfile` exists at repo root
- [ ] `fly.toml` exists at repo root
- [ ] App code exists under `app/` or `src/`
- [ ] README includes build, run, and config steps

## B. fly.toml Contract

- [ ] `app` defined
- [ ] `metadata.networking_mode` defined (`proxy` or `direct`)
- [ ] `internal_port` defined
- [ ] HTTPS behavior configured for service

## C. Runtime Contract

- [ ] App listens on one port
- [ ] Health endpoint exists
- [ ] Startup does not require interactive input

## D. Platform Ingestion

- [ ] `POST /apps/charts/parse` succeeds
- [ ] `POST /apps/from-chart` succeeds (or manual `POST /apps` with parsed metadata)

## E. Provisioning + Routing

- [ ] Pod provisions successfully via `POST /pods/provision`
- [ ] Route resolves via `GET /pods/routing/apps/{slug}`
- [ ] App receives traffic through Proxy/Direct mode as configured

## F. Web App Specific (if browser-facing)

- [ ] App works with cookie-based session (`POST /auth/web-session`)
- [ ] Static asset and API calls work through Data Plane
- [ ] CORS assumptions documented for local/dev

## G. Operational Hygiene

- [ ] No plaintext secrets committed
- [ ] Environment variables documented
- [ ] Failure modes and logs are interpretable
