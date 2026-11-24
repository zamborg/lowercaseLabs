"""Generic upstream Streamable-HTTP MCP proxy provider (`McpUpstreamProvider`).

Connects to any remote Streamable HTTP MCP server using a generic `UpstreamTokenSource`
(e.g. vault-backed OAuth credential), caches upstream tools dynamically, forwards
`tools/call` verbatim (including structuredContent & isError), refreshes on upstream
`notifications/tools/list_changed`, retries once on HTTP 401 after token refresh, and
forwards upstream form elicitation to downstream clients with audit logging.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any
from uuid import UUID

import httpx2
import mcp_types as types
from mcp.client.session import ClientRequestContext, ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client

from zos.auth.context import CallContext
from zos.policy.elicitation import session_supports_elicitation
from zos.providers.base import (
    DownstreamMCPContext,
    ProviderKind,
    ProviderTool,
    UpstreamTokenSource,
    current_downstream_mcp,
)
from zos.router.sessions import SessionTracker
from zos.vault import CredentialRevokedError

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_S = 15.0
DEFAULT_HTTP_READ_TIMEOUT_S = 120.0

ElicitationAuditFn = Callable[
    [CallContext, UUID | None, str, str, str, dict[str, Any]],
    Awaitable[None],
]
HttpClientFactory = Callable[[httpx2.Auth], httpx2.AsyncClient]


class UpstreamAuthFailedError(Exception):
    """Raised when upstream still returns 401 after refreshing the credential once."""


class _UpstreamBearerAuth(httpx2.Auth):
    """httpx2 Auth that injects a bearer token and refreshes once on HTTP 401."""

    def __init__(self, token_source: UpstreamTokenSource) -> None:
        self.token_source = token_source
        self.auth_failed_error: str | None = None

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        self.auth_failed_error = None
        try:
            token = await self.token_source.get_access_token(force_refresh=False)
        except CredentialRevokedError as exc:
            self.auth_failed_error = str(exc)
            raise UpstreamAuthFailedError(
                f"Upstream credential revoked or unavailable. {self.token_source.reconnect_hint}"
            ) from exc

        request.headers["Authorization"] = f"Bearer {token}"
        response = yield request

        if response.status_code == 401:
            # Consume error response body before re-sending the request
            await response.aread()
            try:
                refreshed = await self.token_source.get_access_token(force_refresh=True)
            except CredentialRevokedError as exc:
                self.auth_failed_error = str(exc)
                raise UpstreamAuthFailedError(
                    f"Upstream authentication failed (401). {self.token_source.reconnect_hint}"
                ) from exc

            request.headers["Authorization"] = f"Bearer {refreshed}"
            response2 = yield request
            if response2.status_code == 401:
                self.auth_failed_error = (
                    f"Upstream authentication failed (401) after token refresh. "
                    f"{self.token_source.reconnect_hint}"
                )


def _convert_upstream_tools(mcp_tools: list[types.Tool]) -> list[ProviderTool]:
    """Convert upstream MCP Tool definitions to ProviderTool.

    Note: Upstream annotations are hints, not policy facts. Tier & readOnlyHint/openWorldHint
    always come from ZOS `tool_tiers` + `client_policies`. Omitting `destructive_hint` preserves
    `None` so `build_tool_annotations` applies the MCP spec default (`True` for write/external,
    `False` for read) unless upstream explicitly asserts `False`.
    """
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


class McpUpstreamProvider:
    """Generic Streamable HTTP MCP upstream ToolProvider (kind='upstream')."""

    def __init__(
        self,
        *,
        namespace: str,
        url: str,
        token_source: UpstreamTokenSource,
        sessions: SessionTracker,
        provider_id_getter: Callable[[], UUID | None] | None = None,
        audit_elicitation: ElicitationAuditFn | None = None,
        http_client_factory: HttpClientFactory | None = None,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        transport: str = "mcp",
    ) -> None:
        self._namespace = namespace
        self.url = url
        self.transport = transport
        self.backend = "mcp"
        self.token_source = token_source
        self.sessions = sessions
        self._provider_id_getter = provider_id_getter
        self._audit_elicitation = audit_elicitation
        self._http_client_factory = http_client_factory
        self._connect_timeout_s = connect_timeout_s

        self._session: ClientSession | None = None
        self._auth: _UpstreamBearerAuth | None = None
        self._tools: list[ProviderTool] = []
        self._last_connect_ok = False
        self._last_error: str | None = None

        self._session_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready_event: asyncio.Event | None = None
        self._connect_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()
        self._current_downstream: DownstreamMCPContext | None = None
        self._refresh_tasks: set[asyncio.Task[None]] = set()

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def kind(self) -> ProviderKind:
        return "upstream"

    @property
    def provider_id(self) -> UUID | None:
        return self._provider_id_getter() if self._provider_id_getter is not None else None

    def is_online(self) -> bool:
        """Online iff credential is present and the last connect/list succeeded."""
        return (
            self.token_source.has_credential()
            and self._last_connect_ok
            and self._session is not None
        )

    def _build_http_client(self, auth: httpx2.Auth) -> httpx2.AsyncClient:
        if self._http_client_factory is not None:
            return self._http_client_factory(auth)
        return httpx2.AsyncClient(
            auth=auth,
            timeout=httpx2.Timeout(30.0, read=DEFAULT_HTTP_READ_TIMEOUT_S),
        )

    async def ensure_connected(self, *, broadcast_on_change: bool = True) -> bool:
        """Lazily open (or verify) the long-lived upstream MCP ClientSession."""
        if not self.token_source.has_credential():
            await self.disconnect(broadcast_on_change=broadcast_on_change)
            return False

        async with self._connect_lock:
            if (
                self._session is not None
                and self._last_connect_ok
                and self._session_task is not None
                and not self._session_task.done()
            ):
                return True

            was_online = self._last_connect_ok
            await self._teardown_session_locked()

            stop_event = asyncio.Event()
            ready_event = asyncio.Event()
            self._stop_event = stop_event
            self._ready_event = ready_event
            self._last_error = None

            task = asyncio.create_task(
                self._run_session_loop(stop_event, ready_event),
                name=f"zos-mcp-upstream-{self._namespace}",
            )
            self._session_task = task

            try:
                await asyncio.wait_for(ready_event.wait(), timeout=self._connect_timeout_s)
            except TimeoutError:
                self._last_error = f"Timed out connecting to upstream MCP '{self._namespace}'"
                self._last_connect_ok = False
                await self._teardown_session_locked()

            now_online = self.is_online()
            if broadcast_on_change and now_online != was_online:
                await self.sessions.broadcast_tools_changed()
            return now_online

    async def disconnect(self, *, broadcast_on_change: bool = True) -> None:
        """Tear down any live upstream session and mark offline."""
        async with self._connect_lock:
            was_online = self._last_connect_ok
            await self._teardown_session_locked()
            self._tools = []
            self._last_connect_ok = False
        if broadcast_on_change and was_online:
            await self.sessions.broadcast_tools_changed()

    async def aclose(self) -> None:
        """Clean shutdown hook called from ASGI lifespan exit."""
        for t in list(self._refresh_tasks):
            t.cancel()
        self._refresh_tasks.clear()
        await self.disconnect(broadcast_on_change=False)

    async def _teardown_session_locked(self) -> None:
        stop_ev = self._stop_event
        task = self._session_task
        self._stop_event = None
        self._ready_event = None
        self._session_task = None
        self._session = None
        self._auth = None
        if stop_ev is not None:
            stop_ev.set()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=5.0)

    async def _run_session_loop(
        self,
        stop_event: asyncio.Event,
        ready_event: asyncio.Event,
    ) -> None:
        auth = _UpstreamBearerAuth(self.token_source)
        self._auth = auth
        http_client = self._build_http_client(auth)

        try:
            async with (
                http_client,
                streamable_http_client(self.url, http_client=http_client) as (read_s, write_s),
                ClientSession(
                    read_s,
                    write_s,
                    elicitation_callback=self._on_upstream_elicitation,
                    message_handler=self._on_upstream_message,
                ) as session,
            ):
                await session.initialize()
                tools_res = await session.list_tools()
                self._tools = _convert_upstream_tools(list(tools_res.tools))
                self._session = session
                self._last_connect_ok = True
                self._last_error = None
                logger.info(
                    "Upstream MCP '%s' connected (%s) with %d tool(s)",
                    self._namespace,
                    self.url,
                    len(self._tools),
                )
                ready_event.set()
                await stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err_msg = auth.auth_failed_error or str(exc)
            self._last_error = err_msg
            logger.warning("Upstream MCP '%s' connection error: %s", self._namespace, err_msg)
        finally:
            was_ready = ready_event.is_set()
            was_online = self._last_connect_ok
            self._session = None
            self._last_connect_ok = False
            if not was_ready:
                ready_event.set()
            elif was_online and not stop_event.is_set():
                # Unexpected mid-session drop -> notify downstream clients
                with contextlib.suppress(Exception):
                    await self.sessions.broadcast_tools_changed()

    async def _on_upstream_message(self, message: IncomingMessage) -> None:
        if isinstance(message, types.ToolListChangedNotification):
            logger.info("Upstream MCP '%s' emitted tools/list_changed; refreshing cache", self._namespace)
            task = asyncio.create_task(self._refresh_tools_and_broadcast())
            self._refresh_tasks.add(task)
            task.add_done_callback(self._refresh_tasks.discard)

    async def _refresh_tools_and_broadcast(self) -> None:
        session = self._session
        if session is None:
            return
        try:
            tools_res = await session.list_tools()
            self._tools = _convert_upstream_tools(list(tools_res.tools))
            await self.sessions.broadcast_tools_changed()
        except Exception as exc:
            logger.warning("Failed refreshing tool list from upstream '%s': %s", self._namespace, exc)

    async def _on_upstream_elicitation(
        self,
        context: ClientRequestContext,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Forward upstream form elicitation to downstream client if capable, else decline; audit."""
        downstream = self._current_downstream
        if downstream is None or not session_supports_elicitation(downstream.server_session):
            if downstream is not None and self._audit_elicitation is not None:
                await self._audit_elicitation(
                    downstream.call_ctx,
                    self.provider_id,
                    downstream.exposed_tool,
                    "decline",
                    "denied",
                    {"direction": "upstream", "reason": "client_no_elicitation"},
                )
            return types.ElicitResult(action="decline")

        assert downstream.server_session is not None
        try:
            if isinstance(params, types.ElicitRequestFormParams):
                result = await downstream.server_session.elicit_form(
                    message=params.message,
                    requested_schema=params.requested_schema,
                    related_request_id=downstream.request_id,
                )
            elif isinstance(params, types.ElicitRequestURLParams):
                result = await downstream.server_session.elicit_url(
                    message=params.message,
                    url=params.url,
                    elicitation_id=params.elicitation_id,
                    related_request_id=downstream.request_id,
                )
            else:
                result = types.ElicitResult(action="decline")
        except Exception as exc:
            logger.warning("Downstream elicitation forwarding failed for %s: %s", self._namespace, exc)
            result = types.ElicitResult(action="decline")

        if self._audit_elicitation is not None:
            outcome = "ok" if result.action == "accept" else "denied"
            await self._audit_elicitation(
                downstream.call_ctx,
                self.provider_id,
                downstream.exposed_tool,
                result.action,
                outcome,
                {"direction": "upstream", "action": result.action},
            )
        return result

    async def list_tools(self) -> list[ProviderTool]:
        if not self.token_source.has_credential():
            return []
        if not self.is_online():
            ok = await self.ensure_connected(broadcast_on_change=False)
            if not ok:
                return []
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        if not self.token_source.has_credential():
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"UPSTREAM_OFFLINE: '{self._namespace}' has no credential. {self.token_source.reconnect_hint}",
                    )
                ],
                is_error=True,
            )

        downstream = current_downstream_mcp.get()

        # Serialize calls per upstream provider so upstream elicitation/create correlates 1:1
        # with the exact downstream client & request_id (zero cross-client misrouting).
        async with self._call_lock:
            self._current_downstream = downstream
            try:
                return await self._call_with_reconnect(name, args)
            finally:
                self._current_downstream = None

    async def _call_with_reconnect(
        self,
        name: str,
        args: dict[str, Any],
    ) -> types.CallToolResult:
        for attempt in range(2):
            if not self.is_online():
                connected = await self.ensure_connected(broadcast_on_change=True)
                if not connected:
                    err = self._last_error or f"upstream '{self._namespace}' is unreachable"
                    if "401" in err or "revoked" in err.lower() or "reconnect" in err.lower():
                        return types.CallToolResult(
                            content=[
                                types.TextContent(
                                    type="text",
                                    text=f"UPSTREAM_AUTH_FAILED: {err}",
                                )
                            ],
                            is_error=True,
                        )
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"UPSTREAM_OFFLINE: {err}",
                            )
                        ],
                        is_error=True,
                    )

            session = self._session
            auth = self._auth
            assert session is not None
            try:
                result = await session.call_tool(name, args)
                if isinstance(result, types.CallToolResult):
                    return types.CallToolResult(
                        content=list(result.content),
                        structured_content=result.structured_content,
                        is_error=bool(result.is_error),
                    )
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(result))],
                )
            except Exception as exc:
                # Check if failure was persistent 401 after refresh
                if (auth is not None and auth.auth_failed_error) or isinstance(
                    exc, (UpstreamAuthFailedError, CredentialRevokedError)
                ):
                    msg = (
                        (auth.auth_failed_error if auth and auth.auth_failed_error else str(exc))
                        or f"Upstream authentication failed (401). {self.token_source.reconnect_hint}"
                    )
                    if self.token_source.reconnect_hint not in msg:
                        msg = f"{msg} {self.token_source.reconnect_hint}"
                    await self.disconnect(broadcast_on_change=True)
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=f"UPSTREAM_AUTH_FAILED: {msg}")],
                        is_error=True,
                    )

                logger.warning(
                    "Upstream MCP '%s' call '%s' failed (attempt %d): %s",
                    self._namespace,
                    name,
                    attempt + 1,
                    exc,
                )
                # Tear down dead session; if first attempt, try to reconnect once
                await self.disconnect(broadcast_on_change=(attempt == 1))
                if attempt == 1:
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"UPSTREAM_OFFLINE: upstream '{self._namespace}' call failed ({exc}).",
                            )
                        ],
                        is_error=True,
                    )

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"UPSTREAM_OFFLINE: '{self._namespace}' unavailable.")],
            is_error=True,
        )
