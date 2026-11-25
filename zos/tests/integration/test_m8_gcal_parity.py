"""M8.3 respx tests for gcal REST adapter parity:
get_event, list_calendars, search_events, freebusy, respond_to_event, tier seeds, and scope coverage.
"""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import respx

from tests.conftest import get_test_database_url
from zos.auth.context import CallContext
from zos.config import Settings
from zos.db.queries import V0_TIER_SEEDS, init_principal_and_space
from zos.policy import resolve_tier
from zos.providers.google import GOOGLE_SCOPES, GoogleCalendarProvider, GoogleServiceHub
from zos.router import SessionTracker
from zos.vault import generate_secret_key


async def test_gcal_rest_parity_tools_tiers_and_scopes(db_pool) -> None:
    # Verify scopes cover calendarList, freebusy, events.readonly, events
    assert "https://www.googleapis.com/auth/calendar.calendarlist.readonly" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/calendar.events.freebusy" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/calendar.events.readonly" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/calendar.events" in GOOGLE_SCOPES

    # Verify tiers
    for read_tool in ("gcal_get_event", "gcal_list_calendars", "gcal_search_events", "gcal_freebusy"):
        assert resolve_tier(read_tool, V0_TIER_SEEDS) == "read"
    assert resolve_tier("gcal_respond_to_event", V0_TIER_SEEDS) == "external"

    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url="http://localhost:8787",
        zos_secret_key=generate_secret_key(),
    )
    hub = GoogleServiceHub(db_pool, cfg, SessionTracker())
    await hub.attach_google_tokens(
        account="zubin@example.com",
        scopes=list(GOOGLE_SCOPES),
        access_token="goog-at",
        refresh_token="goog-rt",
        expires_in=3600,
    )
    gcal = GoogleCalendarProvider(hub)
    tool_names = {t.name for t in await gcal.list_tools()}
    assert {
        "list_events",
        "get_event",
        "list_calendars",
        "search_events",
        "freebusy",
        "create_event",
        "update_event",
        "delete_event",
        "respond_to_event",
    } <= tool_names

    ctx = CallContext(
        principal_id=init["principal"]["id"],
        principal_slug="zubin",
        space_id=init["space"]["id"],
        space_slug="main",
        client_id=uuid4(),
        client_name="test-client",
        client_kind="bearer",
        trust_ceiling="owner",
        token_id=uuid4(),
        scopes=("mcp",),
    )

    captured_patch: list[dict] = []

    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://www.googleapis.com/calendar/v3/calendars/primary/events/ev-42").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "ev-42",
                    "summary": "Design Review",
                    "start": {"dateTime": "2026-11-25T10:00:00-08:00"},
                    "end": {"dateTime": "2026-11-25T11:00:00-08:00"},
                    "attendees": [
                        {"email": "peer@example.com", "responseStatus": "accepted"},
                        {"email": "zubin@example.com", "self": True, "responseStatus": "needsAction"},
                    ],
                },
            )
        )
        mock.get("https://www.googleapis.com/calendar/v3/calendars/primary/events/ev-no-self").mock(
            return_value=httpx.Response(
                200,
                json={"id": "ev-no-self", "summary": "Solo", "attendees": [{"email": "other@example.com"}]},
            )
        )

        def patch_rsvp(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            captured_patch.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "ev-42",
                    "summary": "Design Review",
                    "attendees": body["attendees"],
                },
            )

        mock.patch("https://www.googleapis.com/calendar/v3/calendars/primary/events/ev-42").mock(
            side_effect=patch_rsvp
        )

        mock.get("https://www.googleapis.com/calendar/v3/users/me/calendarList").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "primary",
                            "summary": "Zubin",
                            "primary": True,
                            "timeZone": "America/Los_Angeles",
                            "accessRole": "owner",
                        },
                        {
                            "id": "team@group.calendar.google.com",
                            "summary": "Team",
                            "primary": False,
                            "timeZone": "UTC",
                            "accessRole": "reader",
                        },
                    ]
                },
            )
        )

        def search_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params.get("q") == "dentist"
            return httpx.Response(
                200,
                json={"items": [{"id": "ev-d1", "summary": "Dentist Appt"}]},
            )

        mock.get("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
            side_effect=search_handler
        )

        def freebusy_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            assert body["items"] == [{"id": "primary"}, {"id": "team@group.calendar.google.com"}]
            return httpx.Response(
                200,
                json={
                    "timeMin": body["timeMin"],
                    "timeMax": body["timeMax"],
                    "calendars": {
                        "primary": {
                            "busy": [
                                {"start": "2026-11-25T10:00:00-08:00", "end": "2026-11-25T11:00:00-08:00"}
                            ]
                        },
                        "team@group.calendar.google.com": {"busy": []},
                    },
                },
            )

        mock.post("https://www.googleapis.com/calendar/v3/freeBusy").mock(side_effect=freebusy_handler)

        # 1. get_event
        r_get = await gcal.call_tool("get_event", {"event_id": "ev-42"}, ctx)
        assert not r_get.is_error
        ev = json.loads(r_get.content[0].text)
        assert ev["id"] == "ev-42"
        assert ev["my_response_status"] == "needsAction"

        # 2. list_calendars
        r_cals = await gcal.call_tool("list_calendars", {}, ctx)
        assert not r_cals.is_error
        cals = json.loads(r_cals.content[0].text)
        assert cals["count"] == 2
        assert cals["calendars"][0]["primary"] is True

        # 3. search_events
        r_search = await gcal.call_tool(
            "search_events",
            {"query": "dentist", "time_min": "2026-11-01T00:00:00Z"},
            ctx,
        )
        assert not r_search.is_error
        found = json.loads(r_search.content[0].text)
        assert found["count"] == 1
        assert found["events"][0]["id"] == "ev-d1"

        # 4. freebusy (and tz validation)
        r_fb = await gcal.call_tool(
            "freebusy",
            {
                "time_min": "2026-11-25T08:00:00-08:00",
                "time_max": "2026-11-25T18:00:00-08:00",
                "calendar_ids": ["primary", "team@group.calendar.google.com"],
            },
            ctx,
        )
        assert not r_fb.is_error
        fb = json.loads(r_fb.content[0].text)
        assert len(fb["calendars"]["primary"]["busy"]) == 1

        r_fb_bad_tz = await gcal.call_tool(
            "freebusy",
            {"time_min": "2026-11-25T08:00:00", "time_max": "2026-11-25T18:00:00Z"},
            ctx,
        )
        assert r_fb_bad_tz.is_error is True

        # 5. respond_to_event patches ONLY self attendee's responseStatus
        r_rsvp = await gcal.call_tool(
            "respond_to_event",
            {"event_id": "ev-42", "response_status": "accepted", "comment": "See you there"},
            ctx,
        )
        assert not r_rsvp.is_error
        rsvp_out = json.loads(r_rsvp.content[0].text)
        assert rsvp_out["my_response_status"] == "accepted"
        assert len(captured_patch) == 1
        patched_self = next(a for a in captured_patch[0]["attendees"] if a.get("self"))
        assert patched_self["responseStatus"] == "accepted"
        assert patched_self["comment"] == "See you there"

        # Missing self attendee -> clean error
        r_no_self = await gcal.call_tool(
            "respond_to_event",
            {"event_id": "ev-no-self", "response_status": "declined"},
            ctx,
        )
        assert r_no_self.is_error is True
        assert "not listed as an attendee" in r_no_self.content[0].text
