from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from sovereign_store.control_plane.deps import get_current_user, get_db_session
from sovereign_store.models.app import App, NetworkingMode
from sovereign_store.models.user import User
from sovereign_store.schemas.app import (
    AppChartParseRequest,
    AppChartParseResponse,
    AppCreateFromChartRequest,
    AppCreateFromChartResponse,
    AppCreate,
    AppRead,
)
from sovereign_store.services.developer_chart_parser import DeveloperChartParser

router = APIRouter(prefix="/apps", tags=["apps"])
chart_parser = DeveloperChartParser()


def _to_chart_response(result) -> AppChartParseResponse:
    return AppChartParseResponse(
        parse_mode=result.parse_mode,
        repository=result.repository,
        dockerfile_found=result.dockerfile_found,
        fly_toml_found=result.fly_toml_found,
        fly_app_name=result.fly_app_name,
        networking_mode=result.networking_mode,
        internal_port=result.internal_port,
        requires_volume=result.requires_volume,
        warnings=result.warnings,
    )


@router.get("", response_model=list[AppRead])
async def list_apps(session: AsyncSession = Depends(get_db_session)) -> list[AppRead]:
    rows = (await session.execute(select(App).order_by(App.created_at.desc()))).scalars().all()
    return [AppRead.model_validate(app) for app in rows]


@router.post("", response_model=AppRead, status_code=status.HTTP_201_CREATED)
async def create_app(
    payload: AppCreate,
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AppRead:
    app = App(
        slug=payload.slug,
        name=payload.name,
        docker_image=payload.docker_image,
        fly_app_name=payload.fly_app_name,
        networking_mode=payload.networking_mode,
        internal_port=payload.internal_port,
        requires_volume=payload.requires_volume,
    )
    session.add(app)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="App slug already exists",
        ) from exc

    await session.refresh(app)
    return AppRead.model_validate(app)


@router.post("/charts/parse", response_model=AppChartParseResponse)
async def parse_app_chart(
    payload: AppChartParseRequest,
    _: User = Depends(get_current_user),
) -> AppChartParseResponse:
    try:
        result = chart_parser.parse(
            repo_path=payload.repo_path,
            repo_url=str(payload.repo_url) if payload.repo_url else None,
            ref=payload.ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return _to_chart_response(result)


@router.post("/from-chart", response_model=AppCreateFromChartResponse, status_code=status.HTTP_201_CREATED)
async def create_app_from_chart(
    payload: AppCreateFromChartRequest,
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AppCreateFromChartResponse:
    try:
        parsed = chart_parser.parse(
            repo_path=payload.repo_path,
            repo_url=str(payload.repo_url) if payload.repo_url else None,
            ref=payload.ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    warnings = list(parsed.warnings)
    fly_app_name = payload.fly_app_name or parsed.fly_app_name
    if not fly_app_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fly_app_name is required (not found in chart and not provided in payload)",
        )

    networking_mode = payload.networking_mode or parsed.networking_mode
    if not networking_mode:
        networking_mode = NetworkingMode.PROXY
        warnings.append("No networking_mode found; defaulted to proxy")

    internal_port = payload.internal_port or parsed.internal_port or 8080
    requires_volume = (
        payload.requires_volume
        if payload.requires_volume is not None
        else parsed.requires_volume
    )

    app = App(
        slug=payload.slug,
        name=payload.name,
        docker_image=payload.docker_image,
        fly_app_name=fly_app_name,
        networking_mode=networking_mode,
        internal_port=internal_port,
        requires_volume=requires_volume,
    )
    session.add(app)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="App slug already exists",
        ) from exc

    await session.refresh(app)
    parsed.warnings = warnings
    return AppCreateFromChartResponse(
        app=AppRead.model_validate(app),
        parse=_to_chart_response(parsed),
    )
