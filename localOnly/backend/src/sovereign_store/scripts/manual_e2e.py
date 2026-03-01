from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class E2EConfig:
    control_plane_url: str = os.getenv("CONTROL_PLANE_URL", "http://localhost:8000")
    email: str = os.getenv("E2E_EMAIL", "demo@local.dev")
    password: str = os.getenv("E2E_PASSWORD", "password123")
    app_slug: str = os.getenv("E2E_APP_SLUG", "echo-pod")
    app_name: str = os.getenv("E2E_APP_NAME", "Echo Pod")
    docker_image: str = os.getenv("E2E_DOCKER_IMAGE", "replace/me:latest")
    fly_app_name: str = os.getenv("E2E_FLY_APP_NAME", "replace-me-fly-app")
    networking_mode: str = os.getenv("E2E_NETWORKING_MODE", "proxy")
    internal_port: int = int(os.getenv("E2E_INTERNAL_PORT", "8080"))
    requires_volume: bool = os.getenv("E2E_REQUIRES_VOLUME", "false").lower() == "true"


def _raise_for_status(response: httpx.Response, context: str) -> None:
    if response.is_success:
        return
    detail = response.text
    raise RuntimeError(f"{context} failed ({response.status_code}): {detail}")


async def _authenticate(client: httpx.AsyncClient, cfg: E2EConfig) -> dict[str, Any]:
    register_payload = {"email": cfg.email, "password": cfg.password}
    response = await client.post("/auth/register", json=register_payload)

    if response.status_code == 201:
        return response.json()

    if response.status_code == 409:
        login_response = await client.post("/auth/login", json=register_payload)
        _raise_for_status(login_response, "login")
        return login_response.json()

    _raise_for_status(response, "register")
    return {}


async def _ensure_app(client: httpx.AsyncClient, cfg: E2EConfig, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    create_payload = {
        "slug": cfg.app_slug,
        "name": cfg.app_name,
        "docker_image": cfg.docker_image,
        "fly_app_name": cfg.fly_app_name,
        "networking_mode": cfg.networking_mode,
        "internal_port": cfg.internal_port,
        "requires_volume": cfg.requires_volume,
    }
    create_response = await client.post("/apps", json=create_payload, headers=headers)

    if create_response.status_code == 201:
        return create_response.json()

    if create_response.status_code == 409:
        list_response = await client.get("/apps", headers=headers)
        _raise_for_status(list_response, "list apps")
        for app in list_response.json():
            if app["slug"] == cfg.app_slug:
                return app
        raise RuntimeError(f"App slug '{cfg.app_slug}' already exists but was not returned by /apps")

    _raise_for_status(create_response, "create app")
    return {}


async def _provision_pod(client: httpx.AsyncClient, app_id: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.post(
        "/pods/provision",
        json={"app_id": app_id, "member_user_ids": []},
        headers=headers,
    )
    _raise_for_status(response, "provision pod")
    return response.json()


async def _resolve_route(client: httpx.AsyncClient, app_slug: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get(f"/pods/routing/apps/{app_slug}", headers=headers)
    _raise_for_status(response, "resolve route")
    return response.json()


async def run_e2e() -> None:
    cfg = E2EConfig()

    async with httpx.AsyncClient(base_url=cfg.control_plane_url, timeout=30.0) as client:
        auth = await _authenticate(client, cfg)
        token = auth["access_token"]
        app = await _ensure_app(client, cfg, token)
        pod = await _provision_pod(client, app["id"], token)
        route = await _resolve_route(client, cfg.app_slug, token)

    print(
        json.dumps(
            {
                "user": auth["user"],
                "app": {
                    "id": app["id"],
                    "slug": app["slug"],
                    "networking_mode": app["networking_mode"],
                },
                "pod": {
                    "id": pod["id"],
                    "status": pod["status"],
                    "fly_machine_id": pod["fly_machine_id"],
                },
                "route": route,
            },
            indent=2,
        )
    )


def main() -> None:
    asyncio.run(run_e2e())


if __name__ == "__main__":
    main()
