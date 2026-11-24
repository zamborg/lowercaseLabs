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
    return {
        "id": ev.get("id"),
        "summary": ev.get("summary", ""),
        "description": ev.get("description", ""),
        "location": ev.get("location", ""),
        "start": ev.get("start"),
        "end": ev.get("end"),
        "status": ev.get("status"),
        "html_link": ev.get("htmlLink"),
        "attendees": [a.get("email") for a in ev.get("attendees", []) if isinstance(a, dict)],
    }


class GoogleCalendarProvider:
    """Upstream Google Calendar provider sharing the principal's Google credential."""

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
            if name == "create_event":
                return await self._create_event(args)
            if name == "update_event":
                return await self._update_event(args)
            if name == "delete_event":
                return await self._delete_event(args)
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
