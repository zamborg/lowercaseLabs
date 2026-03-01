from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sovereign_store.control_plane.routers.apps import router as apps_router
from sovereign_store.control_plane.routers.auth import router as auth_router
from sovereign_store.control_plane.routers.pods import router as pods_router
from sovereign_store.core.config import get_settings

app = FastAPI(title="Sovereign Control Plane", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(apps_router)
app.include_router(pods_router)


@app.get("/healthz", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run(
        "sovereign_store.control_plane.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
