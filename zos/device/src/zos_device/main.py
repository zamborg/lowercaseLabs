"""Generic ZOS device bridge: relays any stdio MCP server over the /actuator WebSocket.

Usage:
    zos-device --url wss://zos.exe.xyz/actuator --token zos_a_... -- my-mcp-server --arg1
    zos-device --config ~/.zos-actuator/config.toml [-- <command...>]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import shlex
import signal
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import websockets

logger = logging.getLogger("zos_device")

# 16 MiB buffer limit for child stdout lines (default asyncio StreamReader limit is 64 KiB).
STDIO_LINE_LIMIT = 16 * 1024 * 1024
WS_MAX_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Bridge connection parameters and stdio MCP server command."""

    url: str
    token: str
    command: tuple[str, ...]

    @classmethod
    def resolve(
        cls,
        *,
        config_path: Path | None = None,
        url: str | None = None,
        token: str | None = None,
        command: Sequence[str] | None = None,
    ) -> DeviceConfig:
        """Resolve config from TOML file, env vars, and CLI overrides (CLI wins)."""
        raw_toml: dict[str, object] = {}
        if config_path is not None:
            resolved = config_path.expanduser()
            raw_toml = tomllib.loads(resolved.read_text(encoding="utf-8"))

        final_url = (
            (url or "").strip()
            or str(raw_toml.get("url", "")).strip()
            or os.environ.get("ZOS_DEVICE_URL", "").strip()
        )
        final_token = (
            (token or "").strip()
            or str(raw_toml.get("token", "")).strip()
            or os.environ.get("ZOS_DEVICE_TOKEN", "").strip()
        )

        final_cmd: list[str]
        if command:
            final_cmd = [str(part) for part in command]
        else:
            raw_cmd = raw_toml.get("command")
            if isinstance(raw_cmd, list):
                final_cmd = [str(part) for part in raw_cmd if str(part)]
            elif isinstance(raw_cmd, str) and raw_cmd.strip():
                final_cmd = shlex.split(raw_cmd.strip())
            else:
                final_cmd = []

        if not final_url:
            raise ValueError("Missing ZOS actuator WebSocket URL (--url, config 'url', or ZOS_DEVICE_URL)")
        if not final_token:
            raise ValueError("Missing ZOS device token (--token, config 'token', or ZOS_DEVICE_TOKEN)")
        if not final_cmd:
            raise ValueError("Missing stdio MCP server command (-- <command...> or config 'command')")

        return cls(url=final_url, token=final_token, command=tuple(final_cmd))


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    """Gracefully stop child stdio MCP server (close stdin, SIGTERM, fallback SIGKILL)."""
    if proc.stdin is not None and not proc.stdin.is_closing():
        with contextlib.suppress(Exception):
            proc.stdin.close()
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()


def _normalize_ws_to_line(raw: str | bytes) -> bytes | None:
    """Convert an incoming WebSocket JSON-RPC message into a single newline-terminated UTF-8 line."""
    text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    stripped = text.strip()
    if not stripped:
        return None
    if "\n" in stripped or "\r" in stripped:
        stripped = json.dumps(json.loads(stripped), separators=(",", ":"), ensure_ascii=False)
    return (stripped + "\n").encode("utf-8")


async def run_once(
    url: str,
    token: str,
    command: Sequence[str],
    *,
    ping_interval: float = 20.0,
    ping_timeout: float = 10.0,
) -> None:
    """Connect once to ZOS /actuator, spawn the stdio MCP server, and relay until either side ends."""
    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {token}"},
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        max_size=WS_MAX_SIZE,
    ) as ws:
        logger.info("Connected to ZOS device endpoint %s", url)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit bridge stderr for child diagnostics
            limit=STDIO_LINE_LIMIT,
        )
        assert proc.stdin is not None and proc.stdout is not None

        async def ws_to_stdio() -> None:
            async for raw_msg in ws:
                line = _normalize_ws_to_line(raw_msg)
                if line is None:
                    continue
                proc.stdin.write(line)
                await proc.stdin.drain()

        async def stdio_to_ws() -> None:
            while True:
                raw_line = await proc.stdout.readline()
                if not raw_line:
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                await ws.send(text)

        t_in = asyncio.create_task(ws_to_stdio(), name="zos-device-ws-to-stdio")
        t_out = asyncio.create_task(stdio_to_ws(), name="zos-device-stdio-to-ws")
        try:
            done, pending = await asyncio.wait(
                {t_in, t_out},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, websockets.ConnectionClosed):
                    raise exc
        finally:
            for task in (t_in, t_out):
                if not task.done():
                    task.cancel()
            await _terminate_process(proc)


async def run_forever(config: DeviceConfig) -> None:
    """Maintain persistent outbound connection to ZOS with jittered exponential backoff."""
    attempt = 0
    while True:
        started_at = asyncio.get_running_loop().time()
        try:
            await run_once(config.url, config.token, config.command)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Device bridge connection lost: %s", exc)

        elapsed = asyncio.get_running_loop().time() - started_at
        attempt = 0 if elapsed > 10.0 else min(attempt + 1, 6)

        base_delay = min(30.0, float(2 ** max(0, attempt - 1)))
        delay = base_delay * random.uniform(0.75, 1.25)
        logger.info("Reconnecting in %.2fs...", delay)
        await asyncio.sleep(delay)


def parse_args(argv: Sequence[str] | None = None) -> DeviceConfig:
    """Parse CLI arguments (`--url`, `--token`, `--config`, and `-- <command...>`)."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_argv:
        sep = raw_argv.index("--")
        bridge_args = raw_argv[:sep]
        cmd_args = raw_argv[sep + 1 :]
    else:
        bridge_args = raw_argv
        cmd_args = []

    parser = argparse.ArgumentParser(
        prog="zos-device",
        description="Bridge any stdio MCP server to ZOS over the /actuator WebSocket.",
        epilog="Example: zos-device --url wss://zos.exe.xyz/actuator --token zos_a_... -- zos-mac",
    )
    parser.add_argument("--url", help="ZOS /actuator WebSocket URL (ws:// or wss://)")
    parser.add_argument("--token", help="ZOS device token (or set ZOS_DEVICE_TOKEN)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional TOML config file defining url, token, and optionally command",
    )
    ns = parser.parse_args(bridge_args)
    return DeviceConfig.resolve(
        config_path=ns.config,
        url=ns.url,
        token=ns.token,
        command=cmd_args or None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=os.environ.get("ZOS_DEVICE_LOG_LEVEL", "INFO"))
    try:
        cfg = parse_args(argv)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    async def _run() -> None:
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: task.cancel() if task else None)
        with contextlib.suppress(asyncio.CancelledError):
            await run_forever(cfg)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()
