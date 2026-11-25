"""Pluggable `ModuleHost` protocol with `DockerModuleHost` (Unix socket via httpx) and `ManualModuleHost`."""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import quote

import httpx

from zos.catalog import ModuleManifest
from zos.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_SOCKET = "/var/run/docker.sock"
NETWORK_INTERNAL = "zos_modules"
NETWORK_EGRESS = "zos_modules_egress"
DEFAULT_MODULE_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MiB

_BUILD_EXCLUDE_PARTS = frozenset({".venv", "__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"})


@dataclass(frozen=True, slots=True)
class InstalledModule:
    """Result returned by `ModuleHost.install` / `upgrade`."""

    mcp_url: str
    ui_url: str | None
    image_pin: str
    host: Literal["docker", "manual"]
    instructions: str | None = None


@dataclass(frozen=True, slots=True)
class ModuleRuntimeStatus:
    """Live runtime status reported by a `ModuleHost`."""

    host: str
    state: str  # running | stopped | missing | manual | unknown
    container_id: str | None = None
    image_pin: str | None = None
    detail: str | None = None


@runtime_checkable
class ModuleHost(Protocol):
    """Pluggable host backend that runs ZOS modules."""

    kind: Literal["docker", "manual"]

    async def install(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule: ...

    async def uninstall(self, slug: str, *, purge: bool = False) -> None: ...

    async def stop(self, slug: str) -> None: ...

    async def start(self, slug: str) -> None: ...

    async def status(self, slug: str) -> ModuleRuntimeStatus: ...

    async def upgrade(
        self,
        slug: str,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule: ...


def container_name_for(slug: str) -> str:
    return f"zos-mod-{slug}"


def _build_context_tar(build_dir: Path, *, sdk_dir: Path | None = None) -> bytes:
    """Create a deterministic tar archive for Docker Engine `/build`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path in sorted(build_dir.rglob("*")):
            rel = path.relative_to(build_dir)
            if any(part in _BUILD_EXCLUDE_PARTS for part in rel.parts):
                continue
            tf.add(path, arcname=rel.as_posix(), recursive=False)
        # Bundle standalone zos_module_sdk if not already present inside the build dir
        if sdk_dir is not None and sdk_dir.is_dir() and not (build_dir / "zos_module_sdk").exists():
            for fname in ("__init__.py", "caller.py"):
                fpath = sdk_dir / fname
                if fpath.is_file():
                    tf.add(fpath, arcname=f"zos_module_sdk/{fname}", recursive=False)
    return buf.getvalue()


def _resolve_build_context(build_rel: str, repo_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(repo_root / build_rel)
    candidates.extend(
        [
            Path(__file__).resolve().parents[3] / build_rel,
            Path.cwd() / build_rel,
            Path("/app") / build_rel,
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Module build context '{build_rel}' not found (checked: {', '.join(str(c) for c in candidates)})"
    )


def _resolve_sdk_dir(repo_root: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(repo_root / "zos_module_sdk")
    candidates.extend(
        [
            Path(__file__).resolve().parents[3] / "zos_module_sdk",
            Path.cwd() / "zos_module_sdk",
            Path("/app/zos_module_sdk"),
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path
    return None


class DockerModuleHost:
    """Runs ZOS modules via the Docker Engine API over the Unix socket using `httpx`."""

    kind: Literal["docker", "manual"] = "docker"

    def __init__(
        self,
        *,
        docker_socket: str = DEFAULT_DOCKER_SOCKET,
        network_internal: str = NETWORK_INTERNAL,
        network_egress: str = NETWORK_EGRESS,
        memory_limit_bytes: int = DEFAULT_MODULE_MEMORY_BYTES,
        repo_root: Path | None = None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.docker_socket = docker_socket
        self.network_internal = network_internal
        self.network_egress = network_egress
        self.memory_limit_bytes = memory_limit_bytes
        self.repo_root = repo_root
        self._http_client_factory = http_client_factory

    def _client(self, *, timeout_s: float = 60.0) -> httpx.AsyncClient:
        if self._http_client_factory is not None:
            return self._http_client_factory()
        transport = httpx.AsyncHTTPTransport(uds=self.docker_socket)
        return httpx.AsyncClient(
            transport=transport,
            base_url="http://docker",
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

    def build_container_spec(
        self,
        manifest: ModuleManifest,
        *,
        image_ref: str,
        env: dict[str, str],
    ) -> dict[str, Any]:
        """Construct the locked-down Docker container spec (always owned by ZOS, never the manifest)."""
        slug = manifest.slug
        env_list = [f"{k}={v}" for k, v in sorted(env.items())]
        exposed_ports: dict[str, dict[str, Any]] = {f"{manifest.runtime.mcp_port}/tcp": {}}
        if manifest.runtime.ui_port is not None:
            exposed_ports[f"{manifest.runtime.ui_port}/tcp"] = {}

        return {
            "Image": image_ref,
            "Env": env_list,
            "ExposedPorts": exposed_ports,
            "Labels": {
                "zos.module": slug,
                "zos.managed": "true",
                "zos.manifest_version": str(manifest.manifest_version),
                "zos.version": manifest.version,
            },
            "HostConfig": {
                "Privileged": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "Binds": [],
                "Mounts": [],
                "NetworkMode": self.network_internal,
                "PidMode": "",
                "Memory": self.memory_limit_bytes,
                "RestartPolicy": {"Name": "unless-stopped"},
            },
            "NetworkingConfig": {
                "EndpointsConfig": {
                    self.network_internal: {
                        "Aliases": [container_name_for(slug)],
                    }
                }
            },
        }

    async def _ensure_network(self, client: httpx.AsyncClient, name: str, *, internal: bool) -> None:
        resp = await client.get(f"/networks/{quote(name, safe='')}")
        if resp.status_code == 200:
            return
        if resp.status_code != 404:
            raise RuntimeError(f"Docker network check failed for '{name}': {resp.status_code} {resp.text}")
        create = await client.post(
            "/networks/create",
            json={
                "Name": name,
                "Driver": "bridge",
                "Internal": internal,
                "CheckDuplicate": True,
                "Labels": {"zos.managed": "true"},
            },
        )
        if create.status_code not in (200, 201, 409):
            raise RuntimeError(
                f"Failed creating Docker network '{name}': {create.status_code} {create.text}"
            )

    async def _ensure_image(
        self,
        client: httpx.AsyncClient,
        manifest: ModuleManifest,
    ) -> tuple[str, str]:
        """Return `(image_ref_for_create, recorded_image_pin)`."""
        if manifest.runtime.image:
            image_ref = manifest.runtime.image
            inspect = await client.get(f"/images/{quote(image_ref, safe='')}/json")
            if inspect.status_code == 404:
                pull = await client.post(
                    "/images/create",
                    params={"fromImage": image_ref},
                    timeout=300.0,
                )
                if pull.status_code != 200:
                    raise RuntimeError(
                        f"Failed pulling module image '{image_ref}': {pull.status_code} {pull.text}"
                    )
            elif inspect.status_code != 200:
                raise RuntimeError(
                    f"Docker image inspect failed for '{image_ref}': {inspect.status_code} {inspect.text}"
                )
            return image_ref, image_ref

        assert manifest.runtime.build is not None
        build_dir = _resolve_build_context(manifest.runtime.build, repo_root=self.repo_root)
        sdk_dir = _resolve_sdk_dir(repo_root=self.repo_root)
        tar_bytes = _build_context_tar(build_dir, sdk_dir=sdk_dir)
        tag = f"zos-mod-{manifest.slug}:local"
        build_resp = await client.post(
            "/build",
            params={"t": tag, "rm": "true"},
            content=tar_bytes,
            headers={"Content-Type": "application/x-tar"},
            timeout=600.0,
        )
        if build_resp.status_code != 200:
            raise RuntimeError(
                f"Docker build failed for '{manifest.slug}': {build_resp.status_code} {build_resp.text[:500]}"
            )
        # Check streaming JSON lines for build errors (Docker returns HTTP 200 even on build error!)
        for line in build_resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in evt or "errorDetail" in evt:
                detail = evt.get("error") or str(evt.get("errorDetail"))
                raise RuntimeError(f"Docker build failed for '{manifest.slug}': {detail}")

        inspect = await client.get(f"/images/{quote(tag, safe='')}/json")
        if inspect.status_code != 200:
            raise RuntimeError(
                f"Built image '{tag}' not found after build: {inspect.status_code} {inspect.text}"
            )
        image_id = str(inspect.json().get("Id") or tag)
        return tag, image_id

    async def install(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule:
        slug = manifest.slug
        cname = container_name_for(slug)
        full_env = {
            **config,
            **env,
            "ZOS_MODULE_SLUG": slug,
            "MCP_PORT": str(manifest.runtime.mcp_port),
            "MCP_PATH": manifest.runtime.mcp_path,
        }
        if manifest.runtime.ui_port is not None:
            full_env["UI_PORT"] = str(manifest.runtime.ui_port)

        async with self._client(timeout_s=120.0) as client:
            await self._ensure_network(client, self.network_internal, internal=True)
            if manifest.network.egress:
                await self._ensure_network(client, self.network_egress, internal=False)

            image_ref, image_pin = await self._ensure_image(client, manifest)

            # Remove stale container with same name if present
            await client.delete(f"/containers/{quote(cname, safe='')}", params={"force": "true", "v": "true"})

            spec = self.build_container_spec(manifest, image_ref=image_ref, env=full_env)
            create_resp = await client.post(
                "/containers/create",
                params={"name": cname},
                json=spec,
            )
            if create_resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"Docker container create failed for '{slug}': {create_resp.status_code} {create_resp.text}"
                )
            container_id = str(create_resp.json()["Id"])

            if manifest.network.egress:
                conn_resp = await client.post(
                    f"/networks/{quote(self.network_egress, safe='')}/connect",
                    json={"Container": container_id},
                )
                if conn_resp.status_code not in (200, 204):
                    raise RuntimeError(
                        f"Failed attaching '{slug}' to egress network: {conn_resp.status_code} {conn_resp.text}"
                    )

            start_resp = await client.post(f"/containers/{quote(container_id, safe='')}/start")
            if start_resp.status_code not in (204, 304):
                raise RuntimeError(
                    f"Docker container start failed for '{slug}': {start_resp.status_code} {start_resp.text}"
                )

        mcp_url = f"http://{cname}:{manifest.runtime.mcp_port}{manifest.runtime.mcp_path}"
        ui_url = (
            f"http://{cname}:{manifest.runtime.ui_port}"
            if manifest.runtime.ui_port is not None
            else None
        )
        return InstalledModule(
            mcp_url=mcp_url,
            ui_url=ui_url,
            image_pin=image_pin,
            host="docker",
        )

    async def stop(self, slug: str) -> None:
        cname = container_name_for(slug)
        async with self._client(timeout_s=30.0) as client:
            resp = await client.post(f"/containers/{quote(cname, safe='')}/stop", params={"t": "10"})
            if resp.status_code not in (204, 304, 404):
                raise RuntimeError(f"Failed stopping '{cname}': {resp.status_code} {resp.text}")

    async def start(self, slug: str) -> None:
        cname = container_name_for(slug)
        async with self._client(timeout_s=30.0) as client:
            resp = await client.post(f"/containers/{quote(cname, safe='')}/start")
            if resp.status_code not in (204, 304):
                raise RuntimeError(f"Failed starting '{cname}': {resp.status_code} {resp.text}")

    async def uninstall(self, slug: str, *, purge: bool = False) -> None:
        cname = container_name_for(slug)
        async with self._client(timeout_s=30.0) as client:
            await client.post(f"/containers/{quote(cname, safe='')}/stop", params={"t": "5"})
            resp = await client.delete(
                f"/containers/{quote(cname, safe='')}",
                params={"force": "true", "v": "true"},
            )
            if resp.status_code not in (200, 204, 404):
                raise RuntimeError(f"Failed removing '{cname}': {resp.status_code} {resp.text}")

    async def status(self, slug: str) -> ModuleRuntimeStatus:
        cname = container_name_for(slug)
        try:
            async with self._client(timeout_s=10.0) as client:
                resp = await client.get(f"/containers/{quote(cname, safe='')}/json")
        except Exception as exc:
            return ModuleRuntimeStatus(host="docker", state="unknown", detail=str(exc))

        if resp.status_code == 404:
            return ModuleRuntimeStatus(host="docker", state="missing")
        if resp.status_code != 200:
            return ModuleRuntimeStatus(host="docker", state="unknown", detail=f"HTTP {resp.status_code}")

        data = resp.json()
        state_obj = data.get("State") or {}
        running = bool(state_obj.get("Running", False))
        return ModuleRuntimeStatus(
            host="docker",
            state="running" if running else "stopped",
            container_id=str(data.get("Id") or "")[:12] or None,
            image_pin=str(data.get("Image") or "") or None,
            detail=str(state_obj.get("Status") or "") or None,
        )

    async def upgrade(
        self,
        slug: str,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule:
        await self.uninstall(slug, purge=False)
        return await self.install(manifest, config, env=env)


class ManualModuleHost:
    """Fallback host when Docker is unavailable: returns run instructions + `zos module attach`."""

    kind: Literal["docker", "manual"] = "manual"

    def _render_instructions(self, manifest: ModuleManifest) -> str:
        slug = manifest.slug
        cname = container_name_for(slug)
        image_ref = manifest.runtime.image or f"zos-mod-{slug}:local"
        env_flags = [
            f"-e ZOS_MODULE_SLUG={slug}",
            "-e ZOS_MODULE_SECRET=<module-secret-from-zos>",
        ]
        if manifest.storage.postgres:
            env_flags.append("-e ZOS_MODULE_DATABASE_URL=<mod-postgres-url>")
        for field in manifest.config:
            env_flags.append(f"-e {field.name}=<{field.name.lower()}>")

        port_flags = [f"-p 127.0.0.1:{manifest.runtime.mcp_port}:{manifest.runtime.mcp_port}"]
        attach_ui = ""
        if manifest.runtime.ui_port is not None:
            port_flags.append(f"-p 127.0.0.1:{manifest.runtime.ui_port}:{manifest.runtime.ui_port}")
            attach_ui = f" --ui-url http://127.0.0.1:{manifest.runtime.ui_port}"

        build_step = ""
        if manifest.runtime.build:
            build_step = f"docker build -t {image_ref} {manifest.runtime.build}\n"

        docker_cmd = (
            f"{build_step}"
            f"docker run -d --name {cname} --cap-drop ALL --security-opt no-new-privileges "
            f"--memory 256m --restart unless-stopped "
            f"{' '.join(port_flags)} {' '.join(env_flags)} {image_ref}"
        )
        attach_cmd = (
            f"zos module attach {slug} "
            f"--url http://127.0.0.1:{manifest.runtime.mcp_port}{manifest.runtime.mcp_path}"
            f"{attach_ui}"
        )
        return (
            f"Manual install for module '{slug}' (ZOS_MODULE_HOST=manual):\n\n"
            f"1. Start the module container:\n{docker_cmd}\n\n"
            f"2. Register its URL with ZOS:\n{attach_cmd}\n\n"
            f"Or paste this prompt into your coding agent:\n"
            f"  \"Run the ZOS module '{slug}' ({manifest.description}) exposing MCP on port "
            f"{manifest.runtime.mcp_port}{manifest.runtime.mcp_path}"
            + (f" and UI on port {manifest.runtime.ui_port}" if manifest.runtime.ui_port else "")
            + f", then run `{attach_cmd}`.\""
        )

    async def install(
        self,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule:
        pin = manifest.runtime.image or manifest.runtime.build or "manual"
        return InstalledModule(
            mcp_url="",
            ui_url=None,
            image_pin=pin,
            host="manual",
            instructions=self._render_instructions(manifest),
        )

    async def uninstall(self, slug: str, *, purge: bool = False) -> None:
        return None

    async def stop(self, slug: str) -> None:
        return None

    async def start(self, slug: str) -> None:
        return None

    async def status(self, slug: str) -> ModuleRuntimeStatus:
        return ModuleRuntimeStatus(host="manual", state="manual")

    async def upgrade(
        self,
        slug: str,
        manifest: ModuleManifest,
        config: dict[str, str],
        *,
        env: dict[str, str],
    ) -> InstalledModule:
        return await self.install(manifest, config, env=env)


def create_module_host(settings: Settings) -> ModuleHost:
    """Select `DockerModuleHost` or `ManualModuleHost` from `ZOS_MODULE_HOST` and socket presence."""
    mode = settings.module_host.strip().lower()
    sock = settings.docker_socket.strip() or DEFAULT_DOCKER_SOCKET
    if mode == "manual":
        return ManualModuleHost()
    if mode == "docker":
        return DockerModuleHost(
            docker_socket=sock,
            memory_limit_bytes=settings.module_memory_bytes,
        )
    if Path(sock).exists():
        return DockerModuleHost(
            docker_socket=sock,
            memory_limit_bytes=settings.module_memory_bytes,
        )
    return ManualModuleHost()
