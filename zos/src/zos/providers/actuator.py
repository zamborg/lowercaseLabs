"""Core-side WebSocket actuator transport and ToolProvider (`ACTUATOR_OFFLINE`, `ACTUATOR_DISCONNECTED_MID_CALL`)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from uuid import UUID

import anyio
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.shared._context_streams import create_context_streams
from mcp.shared.message import SessionMessage
from mcp_types import jsonrpc_message_adapter
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from zos.audit import record_event
from zos.auth import ActuatorAuthContext, extract_bearer_token, verify_actuator_bearer_token
from zos.auth.context import CallContext
from zos.providers.base import ProviderKind, ProviderTool
from zos.router import ProviderRegistry, SessionTracker

logger = logging.getLogger(__name__)

ACTUATOR_PING_INTERVAL_SECONDS = 20.0
ACTUATOR_PING_TIMEOUT_SECONDS = 10.0


class ActuatorProvider:
    """ToolProvider proxying calls over a live dial-out actuator WebSocket MCP session."""

    def __init__(self, namespace: str, *, provider_id: UUID | None = None) -> None:
        self._namespace = namespace
        self._provider_id = provider_id
        self._session: ClientSession | None = None
        self._tools: list[ProviderTool] = []
        self._online = False
        self._generation = 0
        self._active_ws: WebSocket | None = None
        self._in_flight: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def kind(self) -> ProviderKind:
        return "actuator"

    @property
    def provider_id(self) -> UUID | None:
        return self._provider_id

    def is_online(self) -> bool:
        return self._online and self._session is not None

    async def attach(
        self,
        *,
        provider_id: UUID,
        session: ClientSession,
        tools: list[ProviderTool],
        websocket: WebSocket,
    ) -> int:
        """Attach a newly initialized actuator MCP session, replacing any prior connection."""
        old_ws: WebSocket | None = None
        async with self._lock:
            self._generation += 1
            gen = self._generation
            old_ws = self._active_ws
            self._provider_id = provider_id
            self._session = session
            self._tools = list(tools)
            self._online = True
            self._active_ws = websocket

        if old_ws is not None and old_ws is not websocket:
            with contextlib.suppress(Exception):
                await old_ws.close(code=4000, reason="Replaced by newer actuator connection")
        return gen

    async def detach(self, generation: int) -> bool:
        """Detach if `generation` is still the active connection. Returns True if state changed."""
        async with self._lock:
            if generation != self._generation:
                return False
            was_online = self._online
            self._online = False
            self._session = None
            self._tools = []
            self._active_ws = None
            return was_online

    async def list_tools(self) -> list[ProviderTool]:
        if not self.is_online():
            return []
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        session = self._session
        if not self._online or session is None:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"ACTUATOR_OFFLINE: actuator '{self._namespace}' is not connected.",
                    )
                ],
                is_error=True,
            )

        gen = self._generation
        try:
            result = await session.call_tool(name, args)
            if isinstance(result, types.CallToolResult):
                return result
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
            )
        except Exception as exc:
            # If the actuator disconnected while this call was in flight, signal clearly
            # so callers never blindly retry non-idempotent commands.
            if not self._online or self._generation != gen or self._session is not session:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=(
                                f"ACTUATOR_DISCONNECTED_MID_CALL: actuator '{self._namespace}' "
                                f"disconnected while executing '{name}'. The action may have run; "
                                f"do not retry blindly."
                            ),
                        )
                    ],
                    is_error=True,
                )
            logger.warning("Actuator %s tool %s failed: %s", self._namespace, name, exc)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"ACTUATOR_DISCONNECTED_MID_CALL: actuator '{self._namespace}' call failed ({exc}).",
                    )
                ],
                is_error=True,
            )


def _convert_mcp_tools(mcp_tools: list[types.Tool]) -> list[ProviderTool]:
    converted: list[ProviderTool] = []
    for tool in mcp_tools:
        destructive: bool | None = None
        idempotent: bool | None = None
        if tool.annotations is not None:
            if tool.annotations.destructive_hint is not None:
                destructive = bool(tool.annotations.destructive_hint)
            idempotent = tool.annotations.idempotent_hint
        converted.append(
            ProviderTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema or {"type": "object", "properties": {}}),
                destructive_hint=destructive,
                idempotent_hint=idempotent,
            )
        )
    return converted


class ActuatorManager:
    """Manages /actuator WebSocket connections and their ActuatorProvider instances."""

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
        registry: ProviderRegistry,
        sessions: SessionTracker,
        *,
        ping_interval_s: float = ACTUATOR_PING_INTERVAL_SECONDS,
        ping_timeout_s: float = ACTUATOR_PING_TIMEOUT_SECONDS,
    ) -> None:
        self.pool = pool
        self.registry = registry
        self.sessions = sessions
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self._providers: dict[str, ActuatorProvider] = {}

    def get_or_create_provider(self, slug: str, provider_id: UUID | None = None) -> ActuatorProvider:
        existing = self._providers.get(slug)
        if existing is not None:
            return existing
        provider = ActuatorProvider(slug, provider_id=provider_id)
        self._providers[slug] = provider
        self.registry.register(provider)
        return provider

    async def load_registered_actuators(self) -> None:
        """Pre-register persisted actuator providers as offline so zos_status reports them."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, slug FROM providers WHERE kind = 'actuator' AND enabled = true"
            )
            rows = await cur.fetchall()
        for row in rows:
            self.get_or_create_provider(row["slug"], provider_id=row["id"])

    async def handle_websocket(self, websocket: WebSocket) -> None:
        raw_token = extract_bearer_token(list(websocket.scope.get("headers", [])))
        if not raw_token:
            await websocket.close(code=4401, reason="Missing actuator bearer token")
            return

        auth: ActuatorAuthContext | None = await verify_actuator_bearer_token(self.pool, raw_token)
        if auth is None:
            await websocket.close(code=4401, reason="Invalid or revoked actuator token")
            return

        await websocket.accept()
        provider = self.get_or_create_provider(auth.provider_slug, provider_id=auth.provider_id)

        read_writer, read_stream = create_context_streams[SessionMessage | Exception](32)
        write_stream, write_reader = create_context_streams[SessionMessage](32)
        ws_closed = anyio.Event()
        generation: int | None = None

        async def ws_reader() -> None:
            try:
                while True:
                    text = await websocket.receive_text()
                    msg = jsonrpc_message_adapter.validate_json(text, by_name=False)
                    await read_writer.send(SessionMessage(msg))
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                logger.debug("Actuator %s ws_reader ended: %s", auth.provider_slug, exc)
            finally:
                ws_closed.set()
                with contextlib.suppress(Exception):
                    await read_writer.aclose()

        async def ws_writer() -> None:
            try:
                async for session_msg in write_reader:
                    payload = session_msg.message.model_dump_json(by_alias=True, exclude_none=True)
                    await websocket.send_text(payload)
            except Exception as exc:
                logger.debug("Actuator %s ws_writer ended: %s", auth.provider_slug, exc)
            finally:
                ws_closed.set()
                with contextlib.suppress(Exception):
                    await write_reader.aclose()

        async def ping_loop(session: ClientSession) -> None:
            while not ws_closed.is_set():
                with anyio.move_on_after(self.ping_interval_s):
                    await ws_closed.wait()
                if ws_closed.is_set():
                    return
                with anyio.fail_after(self.ping_timeout_s):
                    await session.send_ping()

        try:
            async with (
                read_writer,
                read_stream,
                write_stream,
                write_reader,
                anyio.create_task_group() as tg,
            ):
                tg.start_soon(ws_reader)
                tg.start_soon(ws_writer)

                async with ClientSession(read_stream, write_stream) as session:
                    with anyio.fail_after(15.0):
                        await session.initialize()
                        tools_result = await session.list_tools()

                    converted = _convert_mcp_tools(list(tools_result.tools))
                    generation = await provider.attach(
                        provider_id=auth.provider_id,
                        session=session,
                        tools=converted,
                        websocket=websocket,
                    )
                    logger.info(
                        "Actuator '%s' connected with %d tool(s)",
                        auth.provider_slug,
                        len(converted),
                    )
                    await record_event(
                        self.pool,
                        principal_id=auth.principal_id,
                        space_id=auth.space_id,
                        provider_id=auth.provider_id,
                        event_type="provider.connected",
                        payload={"slug": auth.provider_slug, "tool_count": len(converted)},
                    )
                    await self.sessions.broadcast_tools_changed()

                    # Keepalive ping loop runs until WS closes or ping times out
                    try:
                        await ping_loop(session)
                    except (TimeoutError, Exception):
                        logger.info("Actuator '%s' ping/connection ended", auth.provider_slug)
                    finally:
                        tg.cancel_scope.cancel()
        except Exception as exc:
            logger.debug("Actuator '%s' session closed: %s", auth.provider_slug, exc)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()
            if generation is not None:
                went_offline = await provider.detach(generation)
                if went_offline:
                    logger.info("Actuator '%s' disconnected", auth.provider_slug)
                    with contextlib.suppress(Exception):
                        await record_event(
                            self.pool,
                            principal_id=auth.principal_id,
                            space_id=auth.space_id,
                            provider_id=auth.provider_id,
                            event_type="provider.disconnected",
                            payload={"slug": auth.provider_slug},
                        )
                    await self.sessions.broadcast_tools_changed()

    def routes(self) -> list[WebSocketRoute]:
        return [WebSocketRoute("/actuator", self.handle_websocket)]
