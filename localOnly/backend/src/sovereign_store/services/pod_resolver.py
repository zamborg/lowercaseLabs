from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sovereign_store.core.config import Settings, get_settings
from sovereign_store.models.app import App, NetworkingMode
from sovereign_store.models.pod import Pod, PodStatus
from sovereign_store.models.pod_member import PodMember


@dataclass(slots=True)
class PodResolution:
    pod_id: UUID
    machine_id: str
    networking_mode: NetworkingMode
    external_ip: str | None
    direct_url: str | None


class PodResolver:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis: Redis | None = None
        if self.settings.redis_url:
            self.redis = Redis.from_url(self.settings.redis_url, encoding="utf-8", decode_responses=True)

    @staticmethod
    def _cache_key(user_id: UUID, app_slug: str) -> str:
        return f"pod-route:{user_id}:{app_slug}"

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    async def resolve_for_user(
        self,
        *,
        session: AsyncSession,
        user_id: UUID,
        app_slug: str,
        pod_access_list: list[UUID] | None = None,
    ) -> PodResolution | None:
        cache_key = self._cache_key(user_id, app_slug)
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                payload = json.loads(cached)
                return PodResolution(
                    pod_id=UUID(payload["pod_id"]),
                    machine_id=payload["machine_id"],
                    networking_mode=NetworkingMode(payload["networking_mode"]),
                    external_ip=payload.get("external_ip"),
                    direct_url=payload.get("direct_url"),
                )

        stmt = (
            select(
                Pod.id,
                Pod.fly_machine_id,
                Pod.external_ip,
                Pod.direct_url,
                App.networking_mode,
            )
            .join(App, App.id == Pod.app_id)
            .join(PodMember, PodMember.pod_id == Pod.id)
            .where(
                App.slug == app_slug,
                PodMember.user_id == user_id,
                Pod.status == PodStatus.ACTIVE,
                Pod.fly_machine_id.is_not(None),
            )
            .limit(1)
        )
        if pod_access_list:
            stmt = stmt.where(Pod.id.in_(pod_access_list))

        row = (await session.execute(stmt)).first()
        if not row:
            return None

        resolution = PodResolution(
            pod_id=row.id,
            machine_id=row.fly_machine_id,
            networking_mode=row.networking_mode,
            external_ip=row.external_ip,
            direct_url=row.direct_url,
        )

        if self.redis:
            await self.redis.setex(
                cache_key,
                30,
                json.dumps(
                    {
                        "pod_id": str(resolution.pod_id),
                        "machine_id": resolution.machine_id,
                        "networking_mode": resolution.networking_mode.value,
                        "external_ip": resolution.external_ip,
                        "direct_url": resolution.direct_url,
                    }
                ),
            )

        return resolution

    async def invalidate(self, *, user_ids: list[UUID], app_slug: str) -> None:
        if not self.redis:
            return
        keys = [self._cache_key(user_id, app_slug) for user_id in user_ids]
        if keys:
            await self.redis.delete(*keys)
