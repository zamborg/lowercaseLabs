# 03: Web App Playbook

This is the required process for generating browser-facing apps on Sovereign.

## 1. Recommended Topology

Use Proxy Mode unless the app has proven direct-mode need.

Benefits:

- Uniform auth gate at Data Plane
- No direct public pod exposure needed
- Simpler rollout and incident handling

## 2. Auth Model For Browsers

Browser requests can authenticate in two ways:

1. `Authorization: Bearer <token>`
2. Web session cookie issued by `POST /auth/web-session`

### Practical rule for web apps

- Use cookie mode for page navigation and static asset requests.
- Use bearer mode for scripted API clients and debugging.

## 3. Frontend Request Rules

- Use same-origin relative paths whenever possible (`/api/...`).
- Use `credentials: "include"` for fetch calls that rely on cookie mode.
- Handle `401` by redirecting users to central auth flow.

## 4. Server-Side Rules

- Serve HTML/JS/CSS over same pod origin.
- Expose health endpoint (`/healthz` or `/api/healthz`).
- Keep startup and routing deterministic.

## 5. Example Structure

```text
web-pod/
  Dockerfile
  fly.toml
  app/
    main.py
    static/
      index.html
```

Reference implementation:
- `test-apps/web-pod`

## 6. Local Verification Flow

1. Authenticate in `web-harness`.
2. Call `POST /auth/web-session`.
3. Resolve route for app slug.
4. Open Data Plane URL `/{app_slug}/` in a new browser tab.
5. Verify page and same-origin API calls function.

## 7. Edge Cases To Handle

- Session expiry and 401 recovery
- Missing cookie scenarios
- Proxy replay returns 202 at gateway layer; app should still serve final content through Fly edge path

## 8. Minimum Acceptance Criteria

- Home page loads in browser through Data Plane route.
- At least one API endpoint callable from frontend.
- Works with cookie mode and bearer mode.
