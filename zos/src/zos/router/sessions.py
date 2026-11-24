"""Live MCP server session tracking for broadcasting notifications/tools/list_changed."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class NotifiableConnection(Protocol):
    async def send_tool_list_changed(self) -> None: ...


class SessionTracker:
    """Tracks live MCP client connections so providers can broadcast tool list changes."""

    def __init__(self) -> None:
        self._connections: set[NotifiableConnection] = set()

    def register(self, connection: NotifiableConnection) -> None:
        self._connections.add(connection)

    def unregister(self, connection: NotifiableConnection) -> None:
        self._connections.discard(connection)

    @property
    def count(self) -> int:
        return len(self._connections)

    async def broadcast_tools_changed(self) -> int:
        """Broadcast notifications/tools/list_changed to all active connections."""
        conns = list(self._connections)
        notified = 0
        for conn in conns:
            try:
                await conn.send_tool_list_changed()
                notified += 1
            except Exception:
                logger.debug("Failed to notify connection of tool list change", exc_info=True)
        return notified
