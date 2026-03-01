from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI(title="Echo Pod", version="0.1.0")


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "pod_id": os.getenv("POD_ID", "unknown"),
        "owner": os.getenv("POD_OWNER_ID", "unknown"),
        "mode": os.getenv("APP_MODE", "unknown"),
    }


@app.api_route("/echo/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def echo(
    path: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    body_bytes = await request.body()
    try:
        body_text = body_bytes.decode("utf-8")
    except UnicodeDecodeError:
        body_text = "<binary>"

    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "path": path,
        "method": request.method,
        "query": dict(request.query_params),
        "body": body_text,
        "pod_id": os.getenv("POD_ID", "unknown"),
        "owner": os.getenv("POD_OWNER_ID", "unknown"),
        "mode": os.getenv("APP_MODE", "unknown"),
    }
