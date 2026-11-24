"""Reusable ephemeral-port ASGI test server helper for E2E tests."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator

import httpx
import uvicorn
from starlette.applications import Starlette


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.asynccontextmanager
async def run_asgi_server(app: Starlette, *, port: int | None = None) -> AsyncIterator[str]:
    """Start a Starlette ASGI app on an ephemeral localhost port and yield its base URL."""
    bound_port = port if port is not None else free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=bound_port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    base_url = f"http://127.0.0.1:{bound_port}"
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                if task.done():
                    exc = task.exception()
                    raise RuntimeError(f"Server exited early: {exc!r}")
                try:
                    r = await client.get(f"{base_url}/healthz", timeout=0.5)
                    if r.status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.05)
            else:
                raise RuntimeError("Server did not become healthy in time")
        yield base_url
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=5.0)
