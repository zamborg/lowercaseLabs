from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_user, issue_access_token
from app.blackhole.api import router as blackhole_router
from app.config import get_settings
from app.db import ensure_storage_directories, get_db
from app.metrics import metrics_payload
from app.models import User
from app.schemas import APIMessage, AuthSession, DevAuthRequest, UserRead


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_storage_directories()
    yield


app = FastAPI(title="blackhole", version="0.1.0", lifespan=lifespan)
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=APIMessage)
def healthcheck() -> APIMessage:
    return APIMessage(message="ok")


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/auth/dev", response_model=AuthSession)
def dev_auth(payload: DevAuthRequest, db: Session = Depends(get_db)) -> AuthSession:
    if not settings.allow_dev_auth:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dev auth is disabled")

    email = payload.email.lower() if payload.email else None
    user = None
    if email:
        user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(display_name=payload.display_name.strip(), email=email, auth_provider="dev")
        db.add(user)
        db.commit()
        db.refresh(user)

    return AuthSession(access_token=issue_access_token(user), user=UserRead.model_validate(user))


@app.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)


app.include_router(blackhole_router)
