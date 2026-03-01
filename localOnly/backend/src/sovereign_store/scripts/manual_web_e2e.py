from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class WebE2EConfig:
    control_plane_url: str = os.getenv("CONTROL_PLANE_URL", "http://localhost:8000")
    email: str = os.getenv("WEB_E2E_EMAIL", "demo@local.dev")
    password: str = os.getenv("WEB_E2E_PASSWORD", "password123")
    app_slug: str = os.getenv("WEB_E2E_APP_SLUG", "web-pod")
    web_probe_path: str = os.getenv("WEB_E2E_PROBE_PATH", "/api/whoami")


def _raise_for_status(response: httpx.Response, context: str) -> None:
    if response.is_success:
        return
    raise RuntimeError(f"{context} failed ({response.status_code}): {response.text}")


async def _authenticate(client: httpx.AsyncClient, cfg: WebE2EConfig) -> dict[str, Any]:
    payload = {"email": cfg.email, "password": cfg.password}
    register = await client.post("/auth/register", json=payload)
    if register.status_code == 201:
        return register.json()
    if register.status_code == 409:
        login = await client.post("/auth/login", json=payload)
        _raise_for_status(login, "login")
        return login.json()
    _raise_for_status(register, "register")
    return {}


async def run_web_e2e() -> None:
    cfg = WebE2EConfig()
    async with httpx.AsyncClient(base_url=cfg.control_plane_url, timeout=30.0) as control:
        auth = await _authenticate(control, cfg)
        token = auth["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        set_session = await control.post(
            "/auth/web-session",
            headers=headers,
        )
        _raise_for_status(set_session, "set web session")

        route_response = await control.get(
            f"/pods/routing/apps/{cfg.app_slug}",
            headers=headers,
        )
        _raise_for_status(route_response, "resolve route")
        route = route_response.json()

        probe_url = f"{route['base_url'].rstrip('/')}{cfg.web_probe_path}"
        probe = await control.get(probe_url)
        if probe.status_code not in {200, 202}:
            raise RuntimeError(
                f"web probe failed ({probe.status_code}): {probe.text}"
            )

    output = {
        "user": auth["user"],
        "resolved_route": route,
        "probe_url": probe_url,
        "probe_status": probe.status_code,
        "probe_body": probe.text,
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    asyncio.run(run_web_e2e())


if __name__ == "__main__":
    main()
