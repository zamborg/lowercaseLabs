"""Google Calendar REST API and upstream MCP backend tools (`gcal_*`) for ZOS Google module."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx2
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from zos_google.oauth import DEFAULT_GCAL_MCP_URL, GoogleClient, GoogleNotConnectedError

VALID_RESPONSE_STATUSES = frozenset({"accepted", "declined", "tentative"})
VALID_BACKENDS = frozenset({"api", "mcp"})


def _validate_tz_aware_iso(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty RFC 3339 timestamp with explicit timezone")
    cleaned = value.strip()
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be valid RFC 3339 with explicit timezone: {value!r}") from exc
    if dt.tzinfo is None:
        raise ValueError(
            f"{field_name} must include an explicit timezone offset (e.g. 'Z' or '-08:00'): {value!r}"
        )
    return cleaned


def _format_event(ev: dict[str, Any]) -> dict[str, Any]:
    raw_attendees = [a for a in ev.get("attendees", []) if isinstance(a, dict)]
    my_status = next((a.get("responseStatus") for a in raw_attendees if a.get("self")), None)
    out: dict[str, Any] = {
        "id": ev.get("id"),
        "summary": ev.get("summary", ""),
        "description": ev.get("description", ""),
        "location": ev.get("location", ""),
        "start": ev.get("start"),
        "end": ev.get("end"),
        "status": ev.get("status"),
        "html_link": ev.get("htmlLink"),
        "attendees": [a.get("email") for a in raw_attendees],
    }
    if my_status is not None:
        out["my_response_status"] = my_status
    return out


GCAL_API_TOOLS: list[types.Tool] = [
    types.Tool(
        name="list_events",
        description="List Google Calendar events in a time range (explicit timezones required).",
        input_schema={
            "type": "object",
            "properties": {
                "time_min": {
                    "type": "string",
                    "description": "RFC 3339 start with timezone (e.g. 2026-11-23T00:00:00-08:00).",
                },
                "time_max": {"type": "string", "description": "RFC 3339 end with timezone."},
                "calendar_id": {"type": "string", "default": "primary"},
                "query": {"type": "string", "description": "Optional free-text filter."},
            },
            "required": ["time_min", "time_max"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    ),
    types.Tool(
        name="get_event",
        description="Fetch a single Google Calendar event by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "calendar_id": {"type": "string", "default": "primary"},
            },
            "required": ["event_id"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    ),
    types.Tool(
        name="list_calendars",
        description="List the user's Google Calendars.",
        input_schema={
            "type": "object",
            "properties": {
                "min_access_role": {
                    "type": "string",
                    "enum": ["freeBusyReader", "owner", "reader", "writer"],
                },
                "show_hidden": {"type": "boolean", "default": False},
            },
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    ),
    types.Tool(
        name="search_events",
        description="Search Google Calendar events by text query (optional RFC 3339 time bounds).",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query."},
                "time_min": {"type": "string", "description": "Optional RFC 3339 start with explicit timezone."},
                "time_max": {"type": "string", "description": "Optional RFC 3339 end with explicit timezone."},
                "calendar_id": {"type": "string", "default": "primary"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            },
            "required": ["query"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    ),
    types.Tool(
        name="freebusy",
        description="Query busy intervals across one or more Google Calendars (explicit timezones required).",
        input_schema={
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "RFC 3339 start with explicit timezone."},
                "time_max": {"type": "string", "description": "RFC 3339 end with explicit timezone."},
                "calendar_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["primary"],
                },
            },
            "required": ["time_min", "time_max"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    ),
    types.Tool(
        name="create_event",
        description="Create a Google Calendar event (explicit timezones required).",
        input_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "start": {"type": "string", "description": "RFC 3339 start with explicit timezone."},
                "end": {"type": "string", "description": "RFC 3339 end with explicit timezone."},
                "calendar_id": {"type": "string", "default": "primary"},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "start", "end"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=True
        ),
    ),
    types.Tool(
        name="update_event",
        description="Update fields on an existing Google Calendar event (explicit timezones required for times).",
        input_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "calendar_id": {"type": "string", "default": "primary"},
                "summary": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "description": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["event_id"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=True
        ),
    ),
    types.Tool(
        name="delete_event",
        description="Delete a Google Calendar event by ID.",
        input_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "calendar_id": {"type": "string", "default": "primary"},
            },
            "required": ["event_id"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False, destructive_hint=True, open_world_hint=True
        ),
    ),
    types.Tool(
        name="respond_to_event",
        description="Update the current user's RSVP responseStatus on a Google Calendar event.",
        input_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "response_status": {
                    "type": "string",
                    "enum": ["accepted", "declined", "tentative"],
                },
                "calendar_id": {"type": "string", "default": "primary"},
                "comment": {"type": "string", "description": "Optional RSVP comment."},
            },
            "required": ["event_id", "response_status"],
        },
        annotations=types.ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        ),
    ),
]


class GoogleCalendarService:
    """Executes Google Calendar tools via either the REST API (`api`) or Google Calendar MCP (`mcp`)."""

    def __init__(
        self,
        client: GoogleClient,
        *,
        default_backend: str = "api",
        mcp_url: str = DEFAULT_GCAL_MCP_URL,
        mcp_http_client_factory: Callable[[str], httpx2.AsyncClient] | None = None,
    ) -> None:
        self.client = client
        clean = default_backend.strip().lower()
        self.default_backend = clean if clean in VALID_BACKENDS else "api"
        self.mcp_url = mcp_url.strip() or DEFAULT_GCAL_MCP_URL
        self._mcp_http_client_factory = mcp_http_client_factory

    async def get_backend(self) -> str:
        saved = await self.client.store.get_setting("gcal_backend")
        if saved in VALID_BACKENDS:
            return saved
        return self.default_backend

    async def set_backend(self, backend: str) -> str:
        clean = backend.strip().lower()
        if clean not in VALID_BACKENDS:
            raise ValueError("Backend must be 'api' or 'mcp'")
        await self.client.store.set_setting("gcal_backend", clean)
        return clean

    def _mcp_http_client(self, token: str) -> httpx2.AsyncClient:
        if self._mcp_http_client_factory is not None:
            return self._mcp_http_client_factory(token)
        return httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})

    async def _call_upstream_mcp_list(self) -> list[types.Tool]:
        try:
            token = await self.client.get_access_token()
        except GoogleNotConnectedError:
            return []

        for attempt in range(2):
            try:
                async with (
                    self._mcp_http_client(token) as hc,
                    streamable_http_client(self.mcp_url, http_client=hc) as (r, w),
                    ClientSession(r, w) as sess,
                ):
                    await sess.initialize()
                    res = await sess.list_tools()
                    return list(res.tools)
            except Exception as exc:
                if attempt == 0 and "401" in str(exc):
                    token = await self.client.get_access_token(force_refresh=True)
                    continue
                raise
        return []

    async def _call_upstream_mcp_tool(self, name: str, args: dict[str, Any]) -> types.CallToolResult:
        token = await self.client.get_access_token()
        for attempt in range(2):
            try:
                async with (
                    self._mcp_http_client(token) as hc,
                    streamable_http_client(self.mcp_url, http_client=hc) as (r, w),
                    ClientSession(r, w) as sess,
                ):
                    await sess.initialize()
                    return await sess.call_tool(name, args)
            except Exception as exc:
                if attempt == 0 and "401" in str(exc):
                    token = await self.client.get_access_token(force_refresh=True)
                    continue
                raise
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="gcal MCP upstream call failed")],
            is_error=True,
        )

    async def list_tools(self) -> list[types.Tool]:
        backend = await self.get_backend()
        if backend == "mcp":
            try:
                return await self._call_upstream_mcp_list()
            except Exception:
                return []
        return list(GCAL_API_TOOLS)

    async def call_tool(self, name: str, args: dict[str, Any]) -> types.CallToolResult:
        backend = await self.get_backend()
        try:
            if backend == "mcp":
                return await self._call_upstream_mcp_tool(name, args)
            if name == "list_events":
                return await self._list_events(args)
            if name == "get_event":
                return await self._get_event(args)
            if name == "list_calendars":
                return await self._list_calendars(args)
            if name == "search_events":
                return await self._search_events(args)
            if name == "freebusy":
                return await self._freebusy(args)
            if name == "create_event":
                return await self._create_event(args)
            if name == "update_event":
                return await self._update_event(args)
            if name == "delete_event":
                return await self._delete_event(args)
            if name == "respond_to_event":
                return await self._respond_to_event(args)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown gcal tool: {name}")],
                is_error=True,
            )
        except Exception as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"gcal_{name} error: {exc}")],
                is_error=True,
            )

    async def _list_events(self, args: dict[str, Any]) -> types.CallToolResult:
        time_min = _validate_tz_aware_iso(args.get("time_min", ""), "time_min")
        time_max = _validate_tz_aware_iso(args.get("time_max", ""), "time_max")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")
        params: dict[str, Any] = {
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": 100,
        }
        if args.get("query"):
            params["q"] = str(args["query"])

        data = await self.client.api_request(
            "GET", f"{self.client.gcal_api_base}/calendars/{cal_id}/events", params=params
        )
        events = [_format_event(ev) for ev in data.get("items", [])]
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"count": len(events), "events": events}, indent=2))]
        )

    async def _get_event(self, args: dict[str, Any]) -> types.CallToolResult:
        event_id = str(args.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("event_id is required")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")
        ev_q = quote(event_id, safe="")
        ev = await self.client.api_request(
            "GET", f"{self.client.gcal_api_base}/calendars/{cal_id}/events/{ev_q}"
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(_format_event(ev), indent=2))]
        )

    async def _list_calendars(self, args: dict[str, Any]) -> types.CallToolResult:
        params: dict[str, Any] = {}
        if args.get("min_access_role"):
            params["minAccessRole"] = str(args["min_access_role"])
        if args.get("show_hidden"):
            params["showHidden"] = "true"
        data = await self.client.api_request(
            "GET", f"{self.client.gcal_api_base}/users/me/calendarList", params=params
        )
        calendars = [
            {
                "id": c.get("id"),
                "summary": c.get("summary", ""),
                "primary": bool(c.get("primary", False)),
                "time_zone": c.get("timeZone", ""),
                "access_role": c.get("accessRole", ""),
            }
            for c in data.get("items", [])
            if isinstance(c, dict)
        ]
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps({"count": len(calendars), "calendars": calendars}, indent=2),
                )
            ]
        )

    async def _search_events(self, args: dict[str, Any]) -> types.CallToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")
        limit = max(1, min(int(args.get("limit", 25)), 100))
        params: dict[str, Any] = {
            "q": query,
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": limit,
        }
        if args.get("time_min"):
            params["timeMin"] = _validate_tz_aware_iso(args["time_min"], "time_min")
        if args.get("time_max"):
            params["timeMax"] = _validate_tz_aware_iso(args["time_max"], "time_max")

        data = await self.client.api_request(
            "GET", f"{self.client.gcal_api_base}/calendars/{cal_id}/events", params=params
        )
        events = [_format_event(ev) for ev in data.get("items", [])]
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"count": len(events), "events": events}, indent=2))]
        )

    async def _freebusy(self, args: dict[str, Any]) -> types.CallToolResult:
        time_min = _validate_tz_aware_iso(args.get("time_min", ""), "time_min")
        time_max = _validate_tz_aware_iso(args.get("time_max", ""), "time_max")
        raw_cals = args.get("calendar_ids") or ["primary"]
        cal_ids = [str(c).strip() for c in raw_cals if str(c).strip()] or ["primary"]

        data = await self.client.api_request(
            "POST",
            f"{self.client.gcal_api_base}/freeBusy",
            json_body={
                "timeMin": time_min,
                "timeMax": time_max,
                "items": [{"id": cid} for cid in cal_ids],
            },
        )
        calendars_out: dict[str, Any] = {}
        for cid, info in (data.get("calendars") or {}).items():
            if isinstance(info, dict):
                calendars_out[cid] = {
                    "busy": list(info.get("busy") or []),
                    "errors": list(info.get("errors") or []),
                }
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "time_min": data.get("timeMin", time_min),
                            "time_max": data.get("timeMax", time_max),
                            "calendars": calendars_out,
                        },
                        indent=2,
                    ),
                )
            ]
        )

    async def _respond_to_event(self, args: dict[str, Any]) -> types.CallToolResult:
        event_id = str(args.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("event_id is required")
        response_status = str(args.get("response_status", "")).strip().lower()
        if response_status not in VALID_RESPONSE_STATUSES:
            raise ValueError("response_status must be one of: accepted, declined, tentative")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")
        ev_q = quote(event_id, safe="")

        current = await self.client.api_request(
            "GET", f"{self.client.gcal_api_base}/calendars/{cal_id}/events/{ev_q}"
        )
        raw_attendees = [dict(a) for a in (current.get("attendees") or []) if isinstance(a, dict)]
        found_self = False
        for att in raw_attendees:
            if att.get("self") is True:
                att["responseStatus"] = response_status
                if args.get("comment") is not None:
                    att["comment"] = str(args["comment"])
                found_self = True
                break
        if not found_self:
            raise ValueError("Current user is not listed as an attendee on this event.")

        updated = await self.client.api_request(
            "PATCH",
            f"{self.client.gcal_api_base}/calendars/{cal_id}/events/{ev_q}",
            json_body={"attendees": raw_attendees},
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(_format_event(updated), indent=2))]
        )

    async def _create_event(self, args: dict[str, Any]) -> types.CallToolResult:
        summary = str(args.get("summary", "")).strip()
        if not summary:
            raise ValueError("summary is required")
        start = _validate_tz_aware_iso(args.get("start", ""), "start")
        end = _validate_tz_aware_iso(args.get("end", ""), "end")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")

        body: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if args.get("description") is not None:
            body["description"] = str(args["description"])
        if args.get("location") is not None:
            body["location"] = str(args["location"])
        if args.get("attendees"):
            body["attendees"] = [{"email": str(e)} for e in args["attendees"]]

        ev = await self.client.api_request(
            "POST", f"{self.client.gcal_api_base}/calendars/{cal_id}/events", json_body=body
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(_format_event(ev), indent=2))]
        )

    async def _update_event(self, args: dict[str, Any]) -> types.CallToolResult:
        event_id = str(args.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("event_id is required")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")
        ev_q = quote(event_id, safe="")

        patch: dict[str, Any] = {}
        if args.get("summary") is not None:
            patch["summary"] = str(args["summary"])
        if args.get("start") is not None:
            patch["start"] = {"dateTime": _validate_tz_aware_iso(args["start"], "start")}
        if args.get("end") is not None:
            patch["end"] = {"dateTime": _validate_tz_aware_iso(args["end"], "end")}
        if args.get("description") is not None:
            patch["description"] = str(args["description"])
        if args.get("location") is not None:
            patch["location"] = str(args["location"])

        ev = await self.client.api_request(
            "PATCH", f"{self.client.gcal_api_base}/calendars/{cal_id}/events/{ev_q}", json_body=patch
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(_format_event(ev), indent=2))]
        )

    async def _delete_event(self, args: dict[str, Any]) -> types.CallToolResult:
        event_id = str(args.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("event_id is required")
        cal_id = quote(str(args.get("calendar_id") or "primary"), safe="")
        ev_q = quote(event_id, safe="")

        await self.client.api_request(
            "DELETE", f"{self.client.gcal_api_base}/calendars/{cal_id}/events/{ev_q}"
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"deleted": True, "event_id": event_id}))]
        )
