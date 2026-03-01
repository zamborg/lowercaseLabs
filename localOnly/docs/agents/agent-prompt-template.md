# Agent Prompt Template

Use this prompt with coding agents to generate compliant app repositories.

---

Build a single-tenant app repository compatible with the Sovereign App Store platform.

Hard requirements:

1. Include `Dockerfile` and `fly.toml` in repo root.
2. Set `[metadata] networking_mode` in `fly.toml` (`proxy` by default unless you justify `direct`).
3. Expose a single HTTP service port (`internal_port`, default 8080).
4. Include health endpoint.
5. Keep runtime fully env-driven and non-interactive.
6. Include a README with build/push/run instructions.

Compatibility target:

- Must pass `POST /apps/charts/parse`.
- Must be registerable via `POST /apps/from-chart`.
- Must be provisionable and routable by Sovereign Control/Data plane.

If building a web app:

- Ensure same-origin browser behavior works through Data Plane route.
- Support cookie-based session flow (`POST /auth/web-session`) and `credentials: include` fetches.

Deliverables:

- Complete app repository tree
- `Dockerfile`
- `fly.toml`
- source code
- README
- explicit list of environment variables

Validation steps to include in your response:

1. build command
2. run command
3. expected health check output
4. any caveats

---
