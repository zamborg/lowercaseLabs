from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

import httpx

from sovereign_store.core.config import Settings, get_settings
from sovereign_store.models.app import App, NetworkingMode


class FlyAPIError(RuntimeError):
    pass


@dataclass(slots=True)
class MachineProvisionResult:
    machine_id: str
    internal_ip: str | None
    external_ip: str | None
    direct_url: str | None


class FlyMachinesClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _volume_name(self, app_slug: str, owner_id: UUID) -> str:
        base = f"{app_slug[:30]}-{str(owner_id).replace('-', '')[:10]}-data"
        normalized = re.sub(r"[^a-z0-9-]", "", base.lower())
        return normalized[:63]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        if not self.settings.fly_api_token:
            raise FlyAPIError("FLY_API_TOKEN is required for provisioning")

        headers = {
            "Authorization": f"Bearer {self.settings.fly_api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(
            base_url=self.settings.fly_api_base_url, timeout=30.0
        ) as client:
            response = await client.request(
                method,
                path,
                headers=headers,
                json=json,
                params=params,
            )

        if response.status_code >= 400:
            raise FlyAPIError(
                f"Fly API call failed ({response.status_code}) {method} {path}: {response.text}"
            )

        if not response.content:
            return {}
        return response.json()

    async def _ensure_volume(self, app_name: str, app_slug: str, owner_id: UUID) -> str:
        volume_name = self._volume_name(app_slug, owner_id)
        payload = {
            "name": volume_name,
            "region": self.settings.fly_default_region,
            "size_gb": self.settings.machine_volume_size_gb,
            "encrypted": True,
        }
        await self._request("POST", f"/v1/apps/{app_name}/volumes", json=payload)
        return volume_name

    async def _ensure_public_ip(self, app_name: str) -> None:
        payload = {"type": "v4"}
        try:
            await self._request("POST", f"/v1/apps/{app_name}/ips", json=payload)
        except FlyAPIError as exc:
            # Duplicate allocation attempts are harmless for direct mode.
            if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
                raise

    async def _first_public_ip(self, app_name: str) -> str | None:
        data = await self._request("GET", f"/v1/apps/{app_name}/ips")
        for item in data.get("ips", []):
            family = str(item.get("family", "")).lower()
            kind = str(item.get("type", "")).lower()
            if kind == "v4" and family in {"public", "global", ""}:
                return item.get("address")
        return None

    async def create_machine(
        self,
        *,
        app: App,
        owner_id: UUID,
        pod_id: UUID,
    ) -> MachineProvisionResult:
        machine_config: dict = {
            "image": app.docker_image,
            "env": {
                "POD_ID": str(pod_id),
                "POD_OWNER_ID": str(owner_id),
                "APP_MODE": app.networking_mode.value,
            },
            "metadata": {
                "pod_id": str(pod_id),
                "owner_id": str(owner_id),
                "app_slug": app.slug,
                "networking_mode": app.networking_mode.value,
            },
            "restart": {"policy": "on-failure", "max_retries": 3},
            "guest": {
                "cpu_kind": self.settings.machine_cpu_kind,
                "cpus": self.settings.machine_cpus,
                "memory_mb": self.settings.machine_memory_mb,
            },
        }

        if app.requires_volume:
            volume_name = await self._ensure_volume(app.fly_app_name, app.slug, owner_id)
            machine_config["mounts"] = [{"volume": volume_name, "path": "/data"}]

        if app.networking_mode == NetworkingMode.DIRECT:
            await self._ensure_public_ip(app.fly_app_name)
            machine_config["services"] = [
                {
                    "protocol": "tcp",
                    "internal_port": app.internal_port,
                    "ports": [{"port": 80, "handlers": ["http"]}],
                }
            ]

        payload = {
            "name": f"pod-{str(pod_id).replace('-', '')[:20]}",
            "region": self.settings.fly_default_region,
            "config": machine_config,
        }

        result = await self._request(
            "POST", f"/v1/apps/{app.fly_app_name}/machines", json=payload
        )
        machine = result.get("machine", result)

        machine_id = machine.get("id")
        if not machine_id:
            raise FlyAPIError(f"Unexpected machine response: {result}")

        internal_ip = machine.get("private_ip") or machine.get("ip")
        external_ip = None
        direct_url = None
        if app.networking_mode == NetworkingMode.DIRECT:
            external_ip = await self._first_public_ip(app.fly_app_name)
            if external_ip:
                direct_url = f"https://{external_ip}"

        return MachineProvisionResult(
            machine_id=machine_id,
            internal_ip=internal_ip,
            external_ip=external_ip,
            direct_url=direct_url,
        )

    async def destroy_machine(self, *, app_name: str, machine_id: str) -> None:
        await self._request(
            "DELETE",
            f"/v1/apps/{app_name}/machines/{machine_id}",
            params={"force": "true"},
        )
