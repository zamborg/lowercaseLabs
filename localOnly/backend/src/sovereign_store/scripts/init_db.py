from __future__ import annotations

import asyncio

from sovereign_store.core.database import Base, engine
from sovereign_store.models import app, pod, pod_member, user  # noqa: F401


async def _init() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def main() -> None:
    asyncio.run(_init())


if __name__ == "__main__":
    main()
