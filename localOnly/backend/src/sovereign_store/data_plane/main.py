from __future__ import annotations

import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from sovereign_store.core.config import get_settings
from sovereign_store.core.database import get_session
from sovereign_store.core.security import decode_access_token
from sovereign_store.services.pod_resolver import PodResolver

app = FastAPI(title="Sovereign Data Plane", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
bearer_scheme = HTTPBearer(auto_error=False)
resolver = PodResolver()


@app.get("/healthz", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.api_route(
    "/{app_slug}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def replay_to_machine(
    request: Request,
    app_slug: str,
    _path: str,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    gateway_token: str | None = Cookie(
        default=None,
        alias=settings.gateway_session_cookie_name,
    ),
    session: AsyncSession = Depends(get_session),
) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    token = credentials.credentials if credentials else gateway_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header or web session cookie is required",
        )

    try:
        token_payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    resolution = await resolver.resolve_for_user(
        session=session,
        user_id=token_payload.user_id,
        app_slug=app_slug,
        pod_access_list=token_payload.pod_access_list,
    )
    if not resolution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active pod found for app '{app_slug}'",
        )

    response = Response(status_code=status.HTTP_202_ACCEPTED)
    response.headers["fly-replay"] = f"instance={resolution.machine_id}"
    response.headers["cache-control"] = "no-store"
    response.headers["vary"] = "Authorization, Cookie"
    return response


@app.on_event("shutdown")
async def close_connections() -> None:
    await resolver.close()


def run() -> None:
    uvicorn.run(
        "sovereign_store.data_plane.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
