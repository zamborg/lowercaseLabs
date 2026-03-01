# Web Pod (Developer Test App)

Minimal web app chart for Sovereign App Store.

## Behavior

- Serves HTML from `/`
- Exposes `/api/whoami`
- Works in Proxy mode with either:
  - Authorization bearer token
  - Web session cookie issued by `POST /auth/web-session`
