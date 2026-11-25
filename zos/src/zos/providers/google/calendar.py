"""Google Calendar upstream ToolProvider (`gcal_*`)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import quote
from uuid import UUID

import mcp_types as types

from zos.auth.context import CallContext
from zos.providers.base import ProviderKind, ProviderTool
from zos.providers.google.client import GoogleServiceHub

GCAL_API_BASE = "https://www.googleapis.com/calendar/v3"


def _validate_tz_aware_iso(value: str, field_name: str) -> str:
    """Require an explicit timezone offset (e.g. 'Z' or '-08:00') on all calendar timestamps."""
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


VALID_RESPONSE_STATUSES = frozenset({"accepted", "declined", "tentative"})


class GoogleCalendarProvider:
    """Upstream Google Calendar REST API provider (`backend='api'`)."""

    backend = "api"

    def __init__(self, hub: GoogleServiceHub) -> None:
        self.hub = hub

    @property
    def namespace(self) -> str:
        return "gcal"

    @property
    def kind(self) -> ProviderKind:
        return "upstream"

    @property
    def provider_id(self) -> UUID | None:
        return self.hub.gcal_provider_id

    def is_online(self) -> bool:
        return self.hub.is_connected

    async def list_tools(self) -> list[ProviderTool]:
        return [
            ProviderTool(
                name="list_events",
                description="List Google Calendar events in a time range (explicit timezones required).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "time_min": {"type": "string", "description": "RFC 3339 start with timezone (e.g. 2026-11-23T00:00:00-08:00)."},
                        "time_max": {"type": "string", "description": "RFC 3339 end with timezone."},
                        "calendar_id": {"type": "string", "default": "primary"},
                        "query": {"type": "string", "description": "Optional free-text filter."},
                    },
                    "required": ["time_min", "time_max"],
                },
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
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
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
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
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
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
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
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
                destructive_hint=False,
                idempotent_hint=True,
            ),
            ProviderTool(
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
                destructive_hint=False,
            ),
            ProviderTool(
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
                destructive_hint=False,
            ),
            ProviderTool(
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
                destructive_hint=True,
            ),
            ProviderTool(
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
                destructive_hint=False,
                idempotent_hint=True,
            ),
        ]

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        ctx: CallContext,
    ) -> types.CallToolResult:
        try:
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

        data = await self.hub.api_request("GET", f"{GCAL_API_BASE}/calendars/{cal_id}/events", params=params)
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
        ev = await self.hub.api_request("GET", f"{GCAL_API_BASE}/calendars/{cal_id}/events/{ev_q}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(_format_event(ev), indent=2))]
        )

    async def _list_calendars(self, args: dict[str, Any]) -> types.CallToolResult:
        params: dict[str, Any] = {}
        if args.get("min_access_role"):
            params["minAccessRole"] = str(args["min_access_role"])
        if args.get("show_hidden"):
            params["showHidden"] = "true"
        data = await self.hub.api_request("GET", f"{GCAL_API_BASE}/users/me/calendarList", params=params)
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

        data = await self.hub.api_request("GET", f"{GCAL_API_BASE}/calendars/{cal_id}/events", params=params)
        events = [_format_event(ev) for ev in data.get("items", [])]
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"count": len(events), "events": events}, indent=2))]
        )

    async def _freebusy(self, args: dict[str, Any]) -> types.CallToolResult:
        time_min = _validate_tz_aware_iso(args.get("time_min", ""), "time_min")
        time_max = _validate_tz_aware_iso(args.get("time_max", ""), "time_max")
        raw_cals = args.get("calendar_ids") or ["primary"]
        cal_ids = [str(c).strip() for c in raw_cals if str(c).strip()]
        if not cal_ids:
            cal_ids = ["primary"]

        data = await self.hub.api_request(
            "POST",
            f"{GCAL_API_BASE}/freeBusy",
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

        current = await self.hub.api_request("GET", f"{GCAL_API_BASE}/calendars/{cal_id}/events/{ev_q}")
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

        updated = await self.hub.api_request(
            "PATCH",
            f"{GCAL_API_BASE}/calendars/{cal_id}/events/{ev_q}",
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

        ev = await self.hub.api_request("POST", f"{GCAL_API_BASE}/calendars/{cal_id}/events", json_body=body)
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

        ev = await self.hub.api_request(
            "PATCH", f"{GCAL_API_BASE}/calendars/{cal_id}/events/{ev_q}", json_body=patch
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

        await self.hub.api_request("DELETE", f"{GCAL_API_BASE}/calendars/{cal_id}/events/{ev_q}")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps({"deleted": True, "event_id": event_id}))]
        )
