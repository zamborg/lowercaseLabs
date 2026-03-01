from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from sovereign_store.control_plane.deps import (
    get_current_user,
    get_db_session,
    get_token_payload,
)
from sovereign_store.core.config import get_settings
from sovereign_store.core.security import TokenPayload
from sovereign_store.models.app import App, NetworkingMode
from sovereign_store.models.pod import Pod, PodStatus
from sovereign_store.models.pod_member import PodMember, PodMemberRole
from sovereign_store.models.user import User
from sovereign_store.schemas.pod import (
    PodMemberAddRequest,
    PodProvisionRequest,
    PodRead,
    PodRouteInfo,
)
from sovereign_store.services.flyio import FlyAPIError, FlyMachinesClient
from sovereign_store.services.pod_resolver import PodResolver

router = APIRouter(prefix="/pods", tags=["pods"])
settings = get_settings()
fly_client = FlyMachinesClient(settings=settings)
pod_resolver = PodResolver(settings=settings)


@router.post("/provision", response_model=PodRead, status_code=status.HTTP_201_CREATED)
async def provision_pod(
    payload: PodProvisionRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PodRead:
    app = (
        await session.execute(select(App).where(App.id == payload.app_id).limit(1))
    ).scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="App not found")

    pod = Pod(app_id=app.id, owner_id=user.id, status=PodStatus.PROVISIONING)
    session.add(pod)
    await session.flush()

    session.add(PodMember(pod_id=pod.id, user_id=user.id, role=PodMemberRole.OWNER))
    for member_id in set(payload.member_user_ids):
        if member_id == user.id:
            continue
        session.add(PodMember(pod_id=pod.id, user_id=member_id, role=PodMemberRole.GUEST))

    try:
        provisioned = await fly_client.create_machine(app=app, owner_id=user.id, pod_id=pod.id)
    except FlyAPIError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provisioning failed: {exc}",
        ) from exc

    pod.fly_machine_id = provisioned.machine_id
    pod.internal_ip = provisioned.internal_ip
    pod.external_ip = provisioned.external_ip
    pod.direct_url = provisioned.direct_url
    pod.status = PodStatus.ACTIVE

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pod could not be saved",
        ) from exc

    await session.refresh(pod)
    await pod_resolver.invalidate(user_ids=[user.id, *payload.member_user_ids], app_slug=app.slug)
    return PodRead.model_validate(pod)


@router.get("", response_model=list[PodRead])
async def list_pods(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[PodRead]:
    rows = (
        await session.execute(
            select(Pod)
            .join(PodMember, PodMember.pod_id == Pod.id)
            .where(PodMember.user_id == user.id)
            .order_by(Pod.created_at.desc())
        )
    ).scalars()
    return [PodRead.model_validate(row) for row in rows]


@router.delete("/{pod_id}", status_code=status.HTTP_204_NO_CONTENT)
async def destroy_pod(
    pod_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    pod = (
        await session.execute(select(Pod).where(Pod.id == pod_id).limit(1))
    ).scalar_one_or_none()
    if not pod:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pod not found")
    if pod.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owner can destroy")

    app = (
        await session.execute(select(App).where(App.id == pod.app_id).limit(1))
    ).scalar_one_or_none()
    if app and pod.fly_machine_id:
        try:
            await fly_client.destroy_machine(app_name=app.fly_app_name, machine_id=pod.fly_machine_id)
        except FlyAPIError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Destroy failed: {exc}",
            ) from exc

    pod.status = PodStatus.SUSPENDED
    await session.commit()
    if app:
        await pod_resolver.invalidate(user_ids=[user.id], app_slug=app.slug)


@router.post("/{pod_id}/members", status_code=status.HTTP_204_NO_CONTENT)
async def add_pod_member(
    pod_id: UUID,
    payload: PodMemberAddRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    pod = (
        await session.execute(select(Pod).where(Pod.id == pod_id).limit(1))
    ).scalar_one_or_none()
    if not pod:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pod not found")
    if pod.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owner can share")

    existing_member = (
        await session.execute(
            select(PodMember)
            .where(PodMember.pod_id == pod_id, PodMember.user_id == payload.user_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing_member:
        return

    session.add(PodMember(pod_id=pod_id, user_id=payload.user_id, role=payload.role))
    await session.commit()

    app = (
        await session.execute(select(App).where(App.id == pod.app_id).limit(1))
    ).scalar_one_or_none()
    if app:
        await pod_resolver.invalidate(user_ids=[payload.user_id, user.id], app_slug=app.slug)


@router.get("/routing/apps/{app_slug}", response_model=PodRouteInfo)
async def resolve_route(
    app_slug: str,
    payload: TokenPayload = Depends(get_token_payload),
    session: AsyncSession = Depends(get_db_session),
) -> PodRouteInfo:
    resolution = await pod_resolver.resolve_for_user(
        session=session,
        user_id=payload.user_id,
        app_slug=app_slug,
        pod_access_list=payload.pod_access_list,
    )
    if not resolution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active pod found for user and app",
        )

    app = (
        await session.execute(select(App).where(App.slug == app_slug).limit(1))
    ).scalar_one_or_none()
    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="App does not exist",
        )

    if app.networking_mode == NetworkingMode.PROXY:
        base_url = f"{settings.edge_router_url.rstrip('/')}/{app.slug}"
    else:
        base_url = resolution.direct_url or (
            f"https://{resolution.external_ip}" if resolution.external_ip else ""
        )
        if not base_url:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Direct mode pod does not have a public route yet",
            )

    return PodRouteInfo(
        app_slug=app.slug,
        networking_mode=app.networking_mode.value,
        pod_id=resolution.pod_id,
        base_url=base_url,
        fly_machine_id=resolution.machine_id,
    )
