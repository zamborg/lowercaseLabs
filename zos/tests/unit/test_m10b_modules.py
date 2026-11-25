"""Unit tests for M10b: manifest v2 validation, caller SDK, and Docker/Manual ModuleHost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from zos_module_sdk.caller import (
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    CallerVerificationError,
    sign_http_caller,
    sign_mcp_caller,
    verify_http_caller,
    verify_mcp_caller,
)

from zos.catalog import (
    ModuleManifest,
    load_catalog,
    load_module_catalog,
    load_module_manifest_file,
)
from zos.modules.host import (
    NETWORK_EGRESS,
    NETWORK_INTERNAL,
    DockerModuleHost,
    ManualModuleHost,
)

DIGEST = "a" * 64


def test_v1_catalog_still_loads_unchanged() -> None:
    entries = load_catalog()
    assert len(entries) == 20
    assert all(e.manifest_version == 1 and e.runtime.kind == "remote" for e in entries)


def test_manifest_v2_validation_digest_and_build(tmp_path: Path) -> None:
    good_img = tmp_path / "demo.toml"
    good_img.write_text(
        f"""
manifest_version = 2
slug = "demo"
name = "Demo Module"
version = "0.1.0"
description = "Demo container module"

[runtime]
kind = "container"
image = "ghcr.io/example/demo@sha256:{DIGEST}"
mcp_port = 8081
mcp_path = "/mcp"
ui_port = 8082

[storage]
postgres = true

[network]
egress = false

[ui]
title = "Demo UI"

[[config]]
name = "API_KEY"
secret = true
help = "Optional API key"

[tiers]
default = "write"
"demo_read" = "read"
""",
        encoding="utf-8",
    )
    m = load_module_manifest_file(good_img)
    assert m.manifest_version == 2
    assert m.slug == "demo"
    assert m.runtime.image == f"ghcr.io/example/demo@sha256:{DIGEST}"
    assert m.runtime.mcp_port == 8081
    assert m.runtime.ui_port == 8082
    assert m.storage.postgres is True
    assert m.network.egress is False
    assert m.ui is not None and m.ui.title == "Demo UI"
    assert len(m.config) == 1 and m.config[0].secret is True
    assert m.tiers.default == "write"
    assert m.tiers.patterns == {"demo_read": "read"}

    # Unpinned tag MUST fail
    bad_tag = tmp_path / "badtag.toml"
    bad_tag.write_text(
        """
manifest_version = 2
slug = "badtag"
name = "Bad Tag"
version = "0.1.0"
description = "Unpinned"

[runtime]
kind = "container"
image = "ghcr.io/example/demo:latest"
mcp_port = 8081
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest-pinned"):
        load_module_manifest_file(bad_tag)

    # Path traversal in build MUST fail
    bad_build = tmp_path / "badbuild.toml"
    bad_build.write_text(
        """
manifest_version = 2
slug = "badbuild"
name = "Bad Build"
version = "0.1.0"
description = "Traversal"

[runtime]
kind = "container"
build = "modules/../etc"
mcp_port = 8081
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative path under 'modules/<slug>'"):
        load_module_manifest_file(bad_build)

    # Both image and build set MUST fail
    both = tmp_path / "both.toml"
    both.write_text(
        f"""
manifest_version = 2
slug = "both"
name = "Both"
version = "0.1.0"
description = "Both set"

[runtime]
kind = "container"
image = "ghcr.io/example/demo@sha256:{DIGEST}"
build = "modules/both"
mcp_port = 8081
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_module_manifest_file(both)

    # load_module_catalog skips malformed files and loads valid ones
    loaded = load_module_catalog(tmp_path)
    assert [x.slug for x in loaded] == ["demo"]


def test_caller_sdk_mcp_sign_and_verify() -> None:
    secret = "test-module-secret-123"
    args = {"to": ["bob"], "subject": "hi", "body": "hello"}
    env = sign_mcp_caller(
        secret,
        principal_id="p-1",
        client_id="c-1",
        client_name="claude",
        tool="send",
        args=args,
        ts=1_700_000_000,
    )
    meta = {MCP_CALLER_META_KEY: env}

    # Valid verification
    verified = verify_mcp_caller(meta, secret=secret, tool="send", args=args, now=1_700_000_010)
    assert verified is not None
    assert verified.principal_id == "p-1"
    assert verified.client_id == "c-1"
    assert verified.client_name == "claude"
    assert verified.ts == 1_700_000_000

    # Unsigned system call returns None
    assert verify_mcp_caller(None, secret=secret, tool="send", args=args) is None
    assert verify_mcp_caller({}, secret=secret, tool="send", args=args) is None

    # Wrong secret -> error
    with pytest.raises(CallerVerificationError):
        verify_mcp_caller(meta, secret="wrong", tool="send", args=args, now=1_700_000_010)

    # Tampered client_name -> error
    tampered = {MCP_CALLER_META_KEY: {"payload": {**env["payload"], "client_name": "owner"}, "sig": env["sig"]}}
    with pytest.raises(CallerVerificationError):
        verify_mcp_caller(tampered, secret=secret, tool="send", args=args, now=1_700_000_010)

    # Wrong tool -> error
    with pytest.raises(CallerVerificationError):
        verify_mcp_caller(meta, secret=secret, tool="archive", args=args, now=1_700_000_010)

    # Mismatched args_digest -> error
    with pytest.raises(CallerVerificationError):
        verify_mcp_caller(meta, secret=secret, tool="send", args={"to": ["eve"]}, now=1_700_000_010)

    # Expired (> +60s) or too far in future (< -60s) -> error
    with pytest.raises(CallerVerificationError, match="skew"):
        verify_mcp_caller(meta, secret=secret, tool="send", args=args, now=1_700_000_061)
    with pytest.raises(CallerVerificationError, match="skew"):
        verify_mcp_caller(meta, secret=secret, tool="send", args=args, now=1_699_999_939)


def test_caller_sdk_http_sign_and_verify() -> None:
    secret = "http-secret-xyz"
    hdr, sig = sign_http_caller(
        secret,
        principal_id="p-1",
        client_id="owner",
        client_name="owner",
        ts=1_700_000_000,
    )
    verified = verify_http_caller(
        {HTTP_CALLER_HEADER: hdr, HTTP_CALLER_SIGNATURE_HEADER: sig},
        secret=secret,
        now=1_700_000_020,
    )
    assert verified.client_name == "owner"
    assert verified.principal_id == "p-1"

    with pytest.raises(CallerVerificationError):
        verify_http_caller(
            {HTTP_CALLER_HEADER: hdr, HTTP_CALLER_SIGNATURE_HEADER: "sha256=deadbeef"},
            secret=secret,
            now=1_700_000_020,
        )
    with pytest.raises(CallerVerificationError, match="skew"):
        verify_http_caller(
            {HTTP_CALLER_HEADER: hdr, HTTP_CALLER_SIGNATURE_HEADER: sig},
            secret=secret,
            now=1_700_000_100,
        )


def _sample_manifest(*, egress: bool = False, build: str | None = None) -> ModuleManifest:
    rt: dict[str, Any] = {"kind": "container", "mcp_port": 8081, "mcp_path": "/mcp", "ui_port": 8082}
    if build:
        rt["build"] = build
    else:
        rt["image"] = f"ghcr.io/example/demo@sha256:{DIGEST}"
    return ModuleManifest.model_validate(
        {
            "manifest_version": 2,
            "slug": "demo",
            "name": "Demo",
            "version": "1.0.0",
            "description": "Demo module",
            "runtime": rt,
            "storage": {"postgres": True},
            "network": {"egress": egress},
            "ui": {"title": "Demo"},
        }
    )


async def test_docker_host_container_spec_hardening_and_networks(tmp_path: Path) -> None:
    calls: list[tuple[str, str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = None
        if request.content and request.headers.get("content-type", "").startswith("application/json"):
            body = json.loads(request.content.decode("utf-8"))
        calls.append((request.method, path, body))
        if request.method == "GET" and path.startswith("/networks/"):
            return httpx.Response(404)
        if request.method == "POST" and path == "/networks/create":
            return httpx.Response(201, json={"Id": "net-1"})
        if request.method == "GET" and path.startswith("/images/"):
            return httpx.Response(200, json={"Id": "sha256:built123"})
        if request.method == "POST" and path == "/build":
            return httpx.Response(200, text='{"stream":"Successfully built"}\n')
        if request.method == "DELETE" and path.startswith("/containers/"):
            return httpx.Response(404)
        if request.method == "POST" and path == "/containers/create":
            return httpx.Response(201, json={"Id": "cid-1234567890ab"})
        if request.method == "POST" and path.endswith("/connect"):
            return httpx.Response(200)
        if request.method == "POST" and path.endswith("/start"):
            return httpx.Response(204)
        if request.method == "GET" and path.endswith("/json"):
            return httpx.Response(200, json={"Id": "cid-1234567890ab", "Image": "sha256:built123", "State": {"Running": True, "Status": "running"}})
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)

    # 1. No-egress image install
    host = DockerModuleHost(
        http_client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://docker")
    )
    manifest = _sample_manifest(egress=False)
    installed = await host.install(
        manifest,
        {},
        env={"ZOS_MODULE_SECRET": "sec", "ZOS_MODULE_DATABASE_URL": "postgresql://mod_demo:x@postgres/zos"},
    )
    assert installed.mcp_url == "http://zos-mod-demo:8081/mcp"
    assert installed.ui_url == "http://zos-mod-demo:8082"
    assert installed.image_pin == f"ghcr.io/example/demo@sha256:{DIGEST}"

    create_call = next(body for method, path, body in calls if method == "POST" and path == "/containers/create")
    hc = create_call["HostConfig"]
    assert hc["Privileged"] is False
    assert hc["CapDrop"] == ["ALL"]
    assert hc["SecurityOpt"] == ["no-new-privileges:true"]
    assert hc["Binds"] == []
    assert hc["Mounts"] == []
    assert hc["NetworkMode"] == NETWORK_INTERNAL
    assert hc["PidMode"] == ""
    assert hc["Memory"] > 0
    assert hc["RestartPolicy"] == {"Name": "unless-stopped"}
    assert create_call["Labels"]["zos.module"] == "demo"
    # No egress network connect call
    assert not any(path == f"/networks/{NETWORK_EGRESS}/connect" for _, path, _ in calls)

    # 2. Egress + local build context
    calls.clear()
    mod_dir = tmp_path / "modules" / "demo"
    mod_dir.mkdir(parents=True)
    (mod_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    host2 = DockerModuleHost(
        repo_root=tmp_path,
        http_client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://docker"),
    )
    installed2 = await host2.install(_sample_manifest(egress=True, build="modules/demo"), {}, env={"ZOS_MODULE_SECRET": "sec"})
    assert installed2.image_pin == "sha256:built123"
    assert any(method == "POST" and path == "/build" for method, path, _ in calls)
    assert any(path == f"/networks/{NETWORK_EGRESS}/connect" for _, path, _ in calls)

    st = await host2.status("demo")
    assert st.state == "running"
    assert st.container_id == "cid-12345678"


async def test_manual_module_host_returns_instructions() -> None:
    host = ManualModuleHost()
    installed = await host.install(_sample_manifest(egress=False), {}, env={"ZOS_MODULE_SECRET": "topsecret"})
    assert installed.host == "manual"
    assert installed.mcp_url == ""
    assert installed.instructions is not None
    assert "docker run" in installed.instructions
    assert "zos module attach demo --url http://127.0.0.1:8081/mcp --ui-url http://127.0.0.1:8082" in installed.instructions
    assert "topsecret" not in installed.instructions
