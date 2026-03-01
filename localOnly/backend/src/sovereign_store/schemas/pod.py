from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from sovereign_store.models.pod import PodStatus
from sovereign_store.models.pod_member import PodMemberRole


class PodProvisionRequest(BaseModel):
    app_id: UUID
    member_user_ids: list[UUID] = Field(default_factory=list)


class PodRead(BaseModel):
    id: UUID
    app_id: UUID
    owner_id: UUID
    fly_machine_id: str | None
    internal_ip: str | None
    external_ip: str | None
    direct_url: str | None
    status: PodStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class PodMemberAddRequest(BaseModel):
    user_id: UUID
    role: PodMemberRole = PodMemberRole.GUEST


class PodRouteInfo(BaseModel):
    app_slug: str
    networking_mode: str
    pod_id: UUID
    base_url: str
    fly_machine_id: str
