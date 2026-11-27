"""Tests for the ZOS Web browser device (install scripts, LaunchAgent rendering, and local E2E)."""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import websockets

WEB_DIR = Path(__file__).resolve().parents[1]
ZOS_ROOT = WEB_DIR.parent.parent
INSTALL_MAC_SH = ZOS_ROOT / "deploy" / "launchd" / "install.sh"
INSTALL_WEB_SH = ZOS_ROOT / "deploy" / "launchd" / "install-web.sh"
DEVICE_SRC = ZOS_ROOT / "device" / "src"

# Pinned @playwright/mcp version and its exact tool/tier inventory.
# Bumping devices/web/VERSION requires re-verifying and updating these expectations.
PINNED_VERSION = "0.0.86"
EXPECTED_READ_TOOLS_0_0_86 = frozenset(
    {
        "browser_console_messages",
        "browser_find",
        "browser_network_request",
        "browser_network_requests",
        "browser_snapshot",
        "browser_take_screenshot",
        "browser_wait_for",
    }
)
EXPECTED_EXTERNAL_TOOLS_0_0_86 = frozenset(
    {
        "browser_click",
        "browser_close",
        "browser_drag",
        "browser_drop",
        "browser_emulate_media",
        "browser_evaluate",
        "browser_file_upload",
        "browser_fill_form",
        "browser_handle_dialog",
        "browser_hover",
        "browser_navigate",
        "browser_navigate_back",
        "browser_press_key",
        "browser_resize",
        "browser_run_code_unsafe",
        "browser_select_option",
        "browser_tabs",
        "browser_type",
    }
)


def _make_stub_node(path: Path, version_str: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho '{version_str}'\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _system_min_path() -> str:
    return "/usr/bin:/bin:/usr/sbin:/sbin"


# ── Part 1: Mac installer Automation -> Messages note ────────────────────────


def test_mac_install_sh_syntax_and_automation_note() -> None:
    res = subprocess.run(["bash", "-n", str(INSTALL_MAC_SH)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    text = INSTALL_MAC_SH.read_text(encoding="utf-8")
    assert "Full Disk Access" in text
    assert "Automation -> Messages" in text
    assert "mac_messages_send" in text
    assert "System Settings -> Privacy & Security -> Automation -> python3.12 -> Messages" in text


# ── Part 2: Web installer & template tests ───────────────────────────────────


def test_web_version_pin_and_install_script_syntax() -> None:
    assert (WEB_DIR / "VERSION").read_text(encoding="utf-8").strip() == PINNED_VERSION
    res = subprocess.run(["bash", "-n", str(INSTALL_WEB_SH)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert os.access(INSTALL_WEB_SH, os.X_OK)


def test_install_web_rejects_old_node_and_missing_node(tmp_path: Path) -> None:
    old_node = _make_stub_node(tmp_path / "old" / "node", "v18.20.4")
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": _system_min_path(),
        "ZOS_WEB_INSTALL_DIR": str(tmp_path / "install"),
        "ZOS_WEB_LAUNCH_AGENTS": str(tmp_path / "la"),
        "ZOS_WEB_LOG_DIR": str(tmp_path / "logs"),
        "ZOS_WEB_SKIP_INSTALL": "1",
    }
    r_old = subprocess.run(
        ["bash", str(INSTALL_WEB_SH), "--node", str(old_node)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r_old.returncode != 0
    assert "Node.js >= 20" in r_old.stderr
    assert "v18.20.4" in r_old.stderr

    r_bad = subprocess.run(
        ["bash", str(INSTALL_WEB_SH), "--node", str(tmp_path / "nonexistent-node")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r_bad.returncode != 0
    assert "not executable" in r_bad.stderr


def test_install_web_resolves_nvm_node_and_renders_isolated_artifacts(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    nvm_dir = fake_home / ".nvm"
    _make_stub_node(nvm_dir / "versions" / "node" / "v18.19.0" / "bin" / "node", "v18.19.0")
    node_22 = _make_stub_node(nvm_dir / "versions" / "node" / "v22.14.0" / "bin" / "node", "v22.14.0")
    _make_stub_node(nvm_dir / "versions" / "node" / "v20.11.1" / "bin" / "node", "v20.11.1")

    install_dir = fake_home / ".zos-web"
    launch_agents = fake_home / "Library" / "LaunchAgents"
    log_dir = fake_home / "Library" / "Logs"

    env = {
        "HOME": str(fake_home),
        "NVM_DIR": str(nvm_dir),
        # Deliberately omit node from PATH so nvm resolution is exercised.
        "PATH": _system_min_path(),
        "ZOS_WEB_INSTALL_DIR": str(install_dir),
        "ZOS_WEB_LAUNCH_AGENTS": str(launch_agents),
        "ZOS_WEB_LOG_DIR": str(log_dir),
        "ZOS_WEB_SKIP_INSTALL": "1",
    }
    res = subprocess.run(
        ["bash", str(INSTALL_WEB_SH)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert f"@playwright/mcp@{PINNED_VERSION}" in res.stdout
    assert "zos device add web" in res.stdout

    # Never touches ~/.zos-actuator or com.zos.actuator
    assert not (fake_home / ".zos-actuator").exists()
    assert not (launch_agents / "com.zos.actuator.plist").exists()

    # Validate rendered plist XML
    plist_path = launch_agents / "com.zos.web.plist"
    assert plist_path.is_file()
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["Label"] == "com.zos.web"
    assert plist["WorkingDirectory"] == str(install_dir / "output")
    argv = plist["ProgramArguments"]
    assert argv[0] == str(install_dir / "venv" / "bin" / "zos-device")
    assert argv[1:4] == ["--config", str(install_dir / "config.toml"), "--"]
    assert argv[4] == str(node_22.resolve())
    assert argv[5] == str(install_dir / "node_modules" / "@playwright" / "mcp" / "cli.js")
    assert "--browser" in argv and argv[argv.index("--browser") + 1] == "chromium"
    assert "--user-data-dir" in argv and argv[argv.index("--user-data-dir") + 1] == str(install_dir / "profile")
    assert "--output-dir" in argv and argv[argv.index("--output-dir") + 1] == str(install_dir / "output")
    assert "--no-webmcp" in argv
    assert "--ignore-roots" in argv
    assert "--config" in argv[5:]
    assert plist["EnvironmentVariables"]["PLAYWRIGHT_BROWSERS_PATH"] == str(install_dir / "browsers")
    assert plist["EnvironmentVariables"]["PATH"].startswith(f"{node_22.resolve().parent}:")

    # Config permissions and defaults (headed + restricted local file access)
    cfg_toml = install_dir / "config.toml"
    assert stat.S_IMODE(cfg_toml.stat().st_mode) == 0o600
    assert stat.S_IMODE((install_dir / "profile").stat().st_mode) == 0o700
    assert stat.S_IMODE((install_dir / "output").stat().st_mode) == 0o700

    pw_cfg = json.loads((install_dir / "playwright-mcp.json").read_text(encoding="utf-8"))
    assert pw_cfg["browser"]["launchOptions"]["headless"] is False
    assert pw_cfg["allowUnrestrictedFileAccess"] is False

    # --headless option creates headless=true on fresh install
    install_dir_hl = fake_home / ".zos-web-hl"
    env_hl = {**env, "ZOS_WEB_INSTALL_DIR": str(install_dir_hl)}
    r_hl = subprocess.run(
        ["bash", str(INSTALL_WEB_SH), "--headless"],
        env=env_hl,
        capture_output=True,
        text=True,
    )
    assert r_hl.returncode == 0, r_hl.stderr
    pw_cfg_hl = json.loads((install_dir_hl / "playwright-mcp.json").read_text(encoding="utf-8"))
    assert pw_cfg_hl["browser"]["launchOptions"]["headless"] is True


# ── Local E2E: zos-device + pinned @playwright/mcp + local static HTML ───────


def _find_e2e_runtime() -> tuple[str, Path, Path] | None:
    """Locate (node_bin, playwright_mcp_cli_js, browsers_dir) if installed for E2E."""
    node_bin = os.environ.get("ZOS_WEB_NODE_BIN") or shutil.which("node")
    if not node_bin:
        return None
    mcp_dir_env = os.environ.get("ZOS_WEB_E2E_MCP_DIR", "").strip()
    if not mcp_dir_env:
        return None
    mcp_dir = Path(mcp_dir_env)
    cli_js = mcp_dir / "node_modules" / "@playwright" / "mcp" / "cli.js"
    browsers_env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    browsers_dir = Path(browsers_env) if browsers_env else (mcp_dir / "browsers")
    if not cli_js.is_file() or not browsers_dir.is_dir():
        return None
    has_chromium = any(p.name.startswith("chromium") for p in browsers_dir.iterdir())
    if not has_chromium:
        return None
    return node_bin, cli_js, browsers_dir


def _zos_default_tier(tool: dict[str, Any]) -> str:
    """Mirror ZOS core device tier derivation: readOnlyHint=True -> 'read', else 'external'."""
    ann = tool.get("annotations") or {}
    return "read" if ann.get("readOnlyHint") is True else "external"


async def test_e2e_zos_device_playwright_mcp_local_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end relay through zos-device -> @playwright/mcp@0.0.86 against a local static HTML page."""
    found = _find_e2e_runtime()
    if found is None:
        pytest.skip(
            "Chromium / @playwright/mcp E2E runtime not configured "
            "(set ZOS_WEB_E2E_MCP_DIR and PLAYWRIGHT_BROWSERS_PATH to run)"
        )
    node_bin, cli_js, browsers_dir = found

    if str(DEVICE_SRC) not in sys.path:
        sys.path.insert(0, str(DEVICE_SRC))
    from zos_device.main import run_once

    # Verify installed @playwright/mcp matches devices/web/VERSION
    pkg = json.loads((cli_js.parent / "package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == PINNED_VERSION
    assert pkg["license"] == "Apache-2.0"

    # Serve a local static HTML page (no external network access)
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "index.html").write_text(
        "<!doctype html><html><head><title>ZOS Web E2E</title></head>"
        "<body><h1>ZOS Web Device</h1><p>Isolated browser verified.</p></body></html>",
        encoding="utf-8",
    )

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return None

    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site_dir))
    )
    http_port = httpd.server_address[1]
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    profile_dir = tmp_path / "profile"
    output_dir = tmp_path / "output"
    profile_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    pw_cfg = tmp_path / "playwright-mcp.json"
    pw_cfg.write_text(
        json.dumps(
            {
                "browser": {"launchOptions": {"headless": True}},
                "allowUnrestrictedFileAccess": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers_dir))
    monkeypatch.chdir(output_dir)

    cmd = [
        node_bin,
        str(cli_js),
        "--browser",
        "chromium",
        "--user-data-dir",
        str(profile_dir),
        "--output-dir",
        str(output_dir),
        "--no-webmcp",
        "--ignore-roots",
        "--config",
        str(pw_cfg),
    ]

    results: dict[str, Any] = {}

    async def recv_id(ws: websockets.ServerConnection, want_id: int) -> dict[str, Any]:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            if msg.get("id") == want_id:
                return msg

    async def ws_handler(ws: websockets.ServerConnection) -> None:
        results["auth"] = ws.request.headers.get("Authorization")
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "zos-test", "version": "0"},
                    },
                }
            )
        )
        results["init"] = await recv_id(ws, 1)
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))

        # 1. List tools and check exact hints-to-tiers mapping for @playwright/mcp@0.0.86
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
        results["tools_list"] = await recv_id(ws, 2)

        # 2. Navigate to local static HTML page
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_navigate",
                        "arguments": {"url": f"http://127.0.0.1:{http_port}/index.html"},
                    },
                }
            )
        )
        results["navigate"] = await recv_id(ws, 3)

        # 3. Take accessibility snapshot
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "browser_snapshot", "arguments": {}},
                }
            )
        )
        results["snapshot"] = await recv_id(ws, 4)

        # 4. Verify file:// URL navigation is blocked when allowUnrestrictedFileAccess=false
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "browser_navigate",
                        "arguments": {"url": f"file://{site_dir / 'index.html'}"},
                    },
                }
            )
        )
        results["file_url"] = await recv_id(ws, 5)

        # 5. Close browser cleanly before disconnecting
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "browser_close", "arguments": {}},
                }
            )
        )
        results["close"] = await recv_id(ws, 6)
        await ws.close()

    try:
        async with websockets.serve(ws_handler, "127.0.0.1", 0) as server:
            ws_port = server.sockets[0].getsockname()[1]
            await run_once(f"ws://127.0.0.1:{ws_port}/actuator", "zos_a_webtest", cmd)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert results["auth"] == "Bearer zos_a_webtest"

    tools = results["tools_list"]["result"]["tools"]
    # Exact 25-tool / 7 read + 18 external inventory for pinned @playwright/mcp@0.0.86
    # (update EXPECTED_* constants if devices/web/VERSION is bumped).
    assert len(tools) == 25
    read_tools = {t["name"] for t in tools if _zos_default_tier(t) == "read"}
    external_tools = {t["name"] for t in tools if _zos_default_tier(t) == "external"}
    assert read_tools == EXPECTED_READ_TOOLS_0_0_86
    assert external_tools == EXPECTED_EXTERNAL_TOOLS_0_0_86
    assert len(read_tools) == 7 and len(external_tools) == 18

    nav_res = results["navigate"]["result"]
    assert not nav_res.get("isError"), nav_res
    snap_res = results["snapshot"]["result"]
    assert not snap_res.get("isError"), snap_res
    snap_text = "\n".join(c.get("text", "") for c in snap_res.get("content", []))
    assert "ZOS Web Device" in snap_text
    assert "Isolated browser verified." in snap_text

    # file:// navigation blocked by default (allowUnrestrictedFileAccess=false)
    file_res = results["file_url"]["result"]
    assert file_res.get("isError") is True

    # Dedicated profile directory was created/populated by Chromium
    assert any(profile_dir.iterdir())
