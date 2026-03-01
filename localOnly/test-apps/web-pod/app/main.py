from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Web Pod", version="0.1.0")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/whoami")
async def whoami(request: Request) -> dict[str, str | bool]:
    return {
        "pod_id": os.getenv("POD_ID", "unknown"),
        "owner": os.getenv("POD_OWNER_ID", "unknown"),
        "mode": os.getenv("APP_MODE", "unknown"),
        "has_auth_header": bool(request.headers.get("authorization")),
        "has_cookie": bool(request.headers.get("cookie")),
    }
