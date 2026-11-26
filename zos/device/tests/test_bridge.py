"""Unit and relay tests for the generic zos-device stdio-to-WebSocket bridge."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import websockets

from zos_device.main import DeviceConfig, parse_args, run_once


def test_config_resolution_cli_toml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Pure CLI with -- command
    cfg = parse_args(
        ["--url", "wss://zos.exe.xyz/actuator", "--token", "zos_a_123", "--", "echo", "hi"]
    )
    assert cfg.url == "wss://zos.exe.xyz/actuator"
    assert cfg.token == "zos_a_123"
    assert cfg.command == ("echo", "hi")

    # 2. TOML file with list command
    toml_path = tmp_path / "device.toml"
    toml_path.write_text(
        'url = "wss://my-zos.exe.xyz/actuator"\n'
        'token = "zos_a_toml"\n'
        'command = ["python3", "-m", "my_server"]\n',
        encoding="utf-8",
    )
    cfg2 = parse_args(["--config", str(toml_path)])
    assert cfg2.url == "wss://my-zos.exe.xyz/actuator"
    assert cfg2.token == "zos_a_toml"
    assert cfg2.command == ("python3", "-m", "my_server")

    # 3. TOML url/token + CLI -- command override (backwards compatible with legacy config.toml)
    legacy_toml = tmp_path / "legacy.toml"
    legacy_toml.write_text(
        'url = "ws://127.0.0.1:8787/actuator"\n'
        'token = "zos_a_legacy"\n',
        encoding="utf-8",
    )
    cfg3 = parse_args(["--config", str(legacy_toml), "--", "zos-mac", "--config", str(legacy_toml)])
    assert cfg3.url == "ws://127.0.0.1:8787/actuator"
    assert cfg3.token == "zos_a_legacy"
    assert cfg3.command == ("zos-mac", "--config", str(legacy_toml))

    # 4. Env var fallback
    monkeypatch.setenv("ZOS_DEVICE_URL", "wss://env.exe.xyz/actuator")
    monkeypatch.setenv("ZOS_DEVICE_TOKEN", "zos_a_env")
    cfg4 = parse_args(["--", "my-server"])
    assert cfg4.url == "wss://env.exe.xyz/actuator"
    assert cfg4.token == "zos_a_env"
    assert cfg4.command == ("my-server",)

    # 5. Missing command fails cleanly
    with pytest.raises(ValueError, match="command"):
        DeviceConfig.resolve(config_path=legacy_toml)


async def test_run_once_relays_jsonrpc_between_ws_and_stdio(tmp_path: Path) -> None:
    """Bridge relays newline-delimited JSON-RPC between a WebSocket server and a child stdio process."""
    echo_script = tmp_path / "echo_stdio.py"
    echo_script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    resp = {'jsonrpc': '2.0', 'id': msg.get('id'), 'result': {'echo': msg.get('method')}}\n"
        "    sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )

    seen_auth: list[str | None] = []
    received: list[dict] = []

    async def ws_handler(websocket: websockets.ServerConnection) -> None:
        seen_auth.append(websocket.request.headers.get("Authorization"))
        await websocket.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
        reply = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        received.append(json.loads(reply))
        await websocket.close()

    async with websockets.serve(ws_handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_once(
            f"ws://127.0.0.1:{port}/actuator",
            "zos_a_testtok",
            [sys.executable, str(echo_script)],
        )

    assert seen_auth == ["Bearer zos_a_testtok"]
    assert received == [{"jsonrpc": "2.0", "id": 1, "result": {"echo": "tools/list"}}]
