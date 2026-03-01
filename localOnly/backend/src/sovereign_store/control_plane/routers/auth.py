from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from sovereign_store.control_plane.deps import (
    get_current_user,
    get_db_session,
)
from sovereign_store.core.config import get_settings
from sovereign_store.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from sovereign_store.models.pod import Pod, PodStatus
from sovereign_store.models.pod_member import PodMember
from sovereign_store.models.user import User
from sovereign_store.schemas.auth import (
    AuthToken,
    LoginRequest,
    RegisterRequest,
    UserRead,
    WebSessionResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


async def _token_for_user(*, session: AsyncSession, user: User) -> AuthToken:
    pod_ids = [
        row[0]
        for row in (
            await session.execute(
                select(Pod.id)
                .join(PodMember, PodMember.pod_id == Pod.id)
                .where(PodMember.user_id == user.id, Pod.status == PodStatus.ACTIVE)
            )
        ).all()
    ]

    token = create_access_token(
        user_id=user.id,
        pod_access_list=pod_ids,
        expires_delta=timedelta(minutes=settings.jwt_exp_minutes),
    )
    return AuthToken(
        access_token=token,
        expires_in=settings.jwt_exp_minutes * 60,
        user=UserRead.model_validate(user),
    )


def _set_gateway_cookie(response: Response, token: str, *, max_age_seconds: int) -> None:
    response.set_cookie(
        key=settings.gateway_session_cookie_name,
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        secure=settings.gateway_session_cookie_secure,
        samesite=settings.gateway_session_cookie_samesite.lower(),
        domain=settings.gateway_session_cookie_domain,
        path=settings.gateway_session_cookie_path,
    )


def _clear_gateway_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.gateway_session_cookie_name,
        domain=settings.gateway_session_cookie_domain,
        path=settings.gateway_session_cookie_path,
    )


@router.post("/register", response_model=AuthToken, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_db_session),
) -> AuthToken:
    user = User(email=payload.email.lower(), password_hash=hash_password(payload.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc

    await session.refresh(user)
    return await _token_for_user(session=session, user=user)


@router.post("/login", response_model=AuthToken)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_db_session),
) -> AuthToken:
    user = (
        await session.execute(select(User).where(User.email == payload.email.lower()).limit(1))
    ).scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    return await _token_for_user(session=session, user=user)


@router.get("/me", response_model=UserRead)
async def me(user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(user)


@router.post("/web-session", response_model=WebSessionResponse)
async def create_web_session(
    response: Response,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> WebSessionResponse:
    auth_token = await _token_for_user(session=session, user=user)
    _set_gateway_cookie(
        response,
        auth_token.access_token,
        max_age_seconds=min(
            auth_token.expires_in, settings.gateway_session_cookie_max_age_seconds
        ),
    )
    return WebSessionResponse(
        active=True,
        cookie_name=settings.gateway_session_cookie_name,
        expires_in=auth_token.expires_in,
    )


@router.delete("/web-session", response_model=WebSessionResponse)
async def clear_web_session(response: Response) -> WebSessionResponse:
    _clear_gateway_cookie(response)
    return WebSessionResponse(
        active=False,
        cookie_name=settings.gateway_session_cookie_name,
        expires_in=None,
    )
