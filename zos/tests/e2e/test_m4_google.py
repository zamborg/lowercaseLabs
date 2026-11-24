"""M4.5 E2E tests: respx-mocked Google OAuth connect, Calendar CRUD, Gmail search/read/send (MIME threading),
429 backoff, 401 refresh, elicitation on gmail_send, and credential revocation taking both providers offline.
"""

from __future__ import annotations

import asyncio
import base64
import email
import json
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
import httpx2
import mcp_types as types
import respx
from argon2 import PasswordHasher
from mcp.client.session import ClientRequestContext, ClientSession, IncomingMessage
from mcp.client.streamable_http import streamable_http_client

from tests.conftest import get_test_database_url
from tests.e2e.helpers import free_port, run_asgi_server
from zos.app import create_app
from zos.config import Settings
from zos.db.queries import create_bearer_client, init_principal_and_space
from zos.vault import generate_secret_key


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


async def test_m4_e2e_google_connect_calendar_gmail_backoff_refresh_and_revocation(db_pool) -> None:
    init = await init_principal_and_space(db_pool, principal_slug="zubin")
    p_id = init["principal"]["id"]
    s_id = init["space"]["id"]

    client_rec = await create_bearer_client(db_pool, principal_id=p_id, space_id=s_id, name="agent-google")
    owner_pass = "google-connect-secret"
    owner_hash = PasswordHasher().hash(owner_pass)

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    mcp_url = f"{base_url}/mcp"
    cfg = Settings(
        database_url=get_test_database_url(),
        zos_base_url=base_url,
        zos_secret_key=generate_secret_key(),
        zos_owner_password_hash=owner_hash,
        google_client_id="fake-google-cid",
        google_client_secret="fake-google-csecret",
    )
    app = create_app(cfg, run_migrations_on_startup=False)
    # Speed up 429 backoff for tests
    app.state.zos.google_hub.backoff_base_s = 0.01

    elicited_tools: list[str] = []

    async def approve_cb(context: ClientRequestContext, params: types.ElicitRequestParams):
        elicited_tools.append(params.message)
        return types.ElicitResult(action="accept", content={"approve": True})

    list_changed_ev = anyio.Event()

    async def on_message(msg: IncomingMessage) -> None:
        if isinstance(msg, types.ToolListChangedNotification):
            list_changed_ev.set()

    sent_payloads: list[dict] = []
    list_events_attempts = 0
    refresh_calls = 0
    in_flight_metadata = 0
    max_concurrent_metadata = 0
    revoke_next_refresh = False

    with respx.mock(assert_all_called=False) as mock:
        # Let localhost traffic pass through to our live test ASGI server
        mock.route(host="127.0.0.1").pass_through()
        mock.route(host="localhost").pass_through()

        # OAuth token endpoint (code exchange + refresh)
        def token_handler(request: httpx.Request) -> httpx.Response:
            nonlocal refresh_calls
            body = parse_qs(request.content.decode("utf-8"))
            gt = body.get("grant_type", [""])[0]
            if gt == "authorization_code":
                return httpx.Response(
                    200,
                    json={
                        "access_token": "goog-at-1",
                        "refresh_token": "goog-rt-1",
                        "expires_in": 3600,
                        "scope": (
                            "https://www.googleapis.com/auth/calendar.events "
                            "https://www.googleapis.com/auth/gmail.readonly "
                            "https://www.googleapis.com/auth/gmail.send"
                        ),
                    },
                )
            refresh_calls += 1
            if revoke_next_refresh:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(
                200,
                json={"access_token": f"goog-at-refreshed-{refresh_calls}", "expires_in": 3600},
            )

        mock.post("https://oauth2.googleapis.com/token").mock(side_effect=token_handler)
        mock.get("https://gmail.googleapis.com/gmail/v1/users/me/profile").mock(
            return_value=httpx.Response(200, json={"emailAddress": "zubin.test@example.com"})
        )

        # Calendar list_events: 429 first, then 200 (tests 429 exponential backoff)
        def gcal_list_handler(request: httpx.Request) -> httpx.Response:
            nonlocal list_events_attempts
            list_events_attempts += 1
            if list_events_attempts == 1:
                return httpx.Response(429, json={"error": {"message": "Rate limit"}})
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "ev-1",
                            "summary": "ZOS Standup",
                            "start": {"dateTime": "2026-11-23T10:00:00-08:00"},
                            "end": {"dateTime": "2026-11-23T10:30:00-08:00"},
                        }
                    ]
                },
            )

        mock.get("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
            side_effect=gcal_list_handler
        )
        mock.post("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "ev-2",
                    "summary": "Dinner",
                    "start": {"dateTime": "2026-11-23T19:00:00-08:00"},
                    "end": {"dateTime": "2026-11-23T20:00:00-08:00"},
                },
            )
        )
        mock.patch("https://www.googleapis.com/calendar/v3/calendars/primary/events/ev-2").mock(
            return_value=httpx.Response(200, json={"id": "ev-2", "summary": "Dinner Updated"})
        )
        mock.delete("https://www.googleapis.com/calendar/v3/calendars/primary/events/ev-2").mock(
            return_value=httpx.Response(204)
        )

        # Gmail search + metadata concurrency check (8 messages -> concurrency <= 5)
        mock.get("https://gmail.googleapis.com/gmail/v1/users/me/messages").mock(
            return_value=httpx.Response(
                200,
                json={"messages": [{"id": f"m{i}", "threadId": f"t{i}"} for i in range(8)]},
            )
        )

        async def gmail_get_msg(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight_metadata, max_concurrent_metadata
            mid = request.url.path.rsplit("/", 1)[-1]
            fmt = request.url.params.get("format", "full")

            if fmt == "metadata":
                in_flight_metadata += 1
                max_concurrent_metadata = max(max_concurrent_metadata, in_flight_metadata)
                await asyncio.sleep(0.02)
                in_flight_metadata -= 1
                return httpx.Response(
                    200,
                    json={
                        "id": mid,
                        "threadId": f"thread-{mid}",
                        "snippet": f"Snippet for {mid}",
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "alice@example.com"},
                                {"name": "To", "value": "zubin.test@example.com"},
                                {"name": "Subject", "value": f"Hello {mid}"},
                                {"name": "Date", "value": "Mon, 23 Nov 2026 10:00:00 -0800"},
                                {"name": "Message-ID", "value": f"<{mid}@example.com>"},
                                {"name": "References", "value": "<root@example.com>"},
                            ]
                        },
                    },
                )

            if mid == "m-html":
                return httpx.Response(
                    200,
                    json={
                        "id": "m-html",
                        "threadId": "thread-html",
                        "payload": {
                            "mimeType": "text/html",
                            "headers": [{"name": "Subject", "value": "HTML Only"}],
                            "body": {
                                "data": _b64url(
                                    "<html><head><style>body{color:red}</style></head>"
                                    "<body><script>evil()</script><p>Hello <b>World</b></p></body></html>"
                                )
                            },
                        },
                    },
                )

            if mid == "m-401-then-ok":
                if request.headers.get("Authorization") == "Bearer goog-at-1":
                    return httpx.Response(401, json={"error": {"message": "Invalid Credentials"}})
                return httpx.Response(
                    200,
                    json={
                        "id": "m-401-then-ok",
                        "threadId": "t-ok",
                        "payload": {
                            "mimeType": "text/plain",
                            "headers": [{"name": "Subject", "value": "Refreshed"}],
                            "body": {"data": _b64url("Worked after refresh")},
                        },
                    },
                )

            if mid == "m-revoke":
                return httpx.Response(401, json={"error": {"message": "Token revoked"}})

            return httpx.Response(
                200,
                json={
                    "id": mid,
                    "threadId": f"thread-{mid}",
                    "payload": {
                        "mimeType": "multipart/alternative",
                        "headers": [{"name": "Subject", "value": "Plain Preferred"}],
                        "parts": [
                            {"mimeType": "text/plain", "body": {"data": _b64url("Plain body text")}},
                            {"mimeType": "text/html", "body": {"data": _b64url("<p>HTML ignored</p>")}},
                        ],
                    },
                },
            )

        mock.get(url__regex=r"^https://gmail\.googleapis\.com/gmail/v1/users/me/messages/[^/]+$").mock(
            side_effect=gmail_get_msg
        )

        def gmail_send_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            sent_payloads.append(payload)
            return httpx.Response(200, json={"id": "sent-1", "threadId": payload.get("threadId", "t-new")})

        mock.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send").mock(
            side_effect=gmail_send_handler
        )

        async with run_asgi_server(app, port=port):
            http_mcp = httpx2.AsyncClient(headers={"Authorization": f"Bearer {client_rec['token']}"})
            async with (
                http_mcp,
                streamable_http_client(mcp_url, http_client=http_mcp) as (rs, ws),
                ClientSession(rs, ws, elicitation_callback=approve_cb, message_handler=on_message) as session,
            ):
                await session.initialize()
                await asyncio.sleep(0.2)

                # 1. Before connect: no gcal_* or gmail_* tools
                tools_pre = {t.name for t in (await session.list_tools()).tools}
                assert not any(n.startswith(("gcal_", "gmail_")) for n in tools_pre)

                # 2. Connect Google via /connect/google + callback
                list_changed_ev = anyio.Event()
                async with httpx.AsyncClient(follow_redirects=False) as web:
                    r_bad = await web.post(f"{base_url}/connect/google", data={"passphrase": "wrong"})
                    assert r_bad.status_code == 401

                    r_go = await web.post(f"{base_url}/connect/google", data={"passphrase": owner_pass})
                    assert r_go.status_code == 302
                    loc = r_go.headers["location"]
                    qs = parse_qs(urlparse(loc).query)
                    assert qs["access_type"] == ["offline"]
                    assert qs["prompt"] == ["consent"]
                    state = qs["state"][0]

                    r_cb = await web.get(f"{base_url}/connect/google/callback", params={"code": "g-code", "state": state})
                    assert r_cb.status_code == 200

                with anyio.fail_after(5):
                    await list_changed_ev.wait()

                # Verify single shared credential row serves both gcal and gmail providers
                async with db_pool.connection() as conn:
                    creds = await (await conn.execute("SELECT id FROM credentials")).fetchall()
                    provs = await (
                        await conn.execute("SELECT slug, credential_id FROM providers WHERE slug IN ('gcal', 'gmail')")
                    ).fetchall()
                    assert len(creds) == 1
                    assert len(provs) == 2
                    assert all(pr["credential_id"] == creds[0]["id"] for pr in provs)

                tools = {t.name: t for t in (await session.list_tools()).tools}
                for expected_tool in (
                    "gcal_list_events",
                    "gcal_create_event",
                    "gcal_update_event",
                    "gcal_delete_event",
                    "gmail_search",
                    "gmail_read",
                    "gmail_send",
                ):
                    assert expected_tool in tools
                assert tools["gcal_delete_event"].annotations.destructive_hint is True
                assert tools["gcal_list_events"].annotations.read_only_hint is True
                assert tools["gmail_send"].annotations.open_world_hint is True

                # 3. Calendar: rejects naive datetime; succeeds with explicit TZ and 429 retry
                bad_tz = await session.call_tool(
                    "gcal_list_events",
                    {"time_min": "2026-11-23T00:00:00", "time_max": "2026-11-24T00:00:00"},
                )
                assert bad_tz.is_error
                assert "timezone" in bad_tz.content[0].text.lower()

                ev_list = await session.call_tool(
                    "gcal_list_events",
                    {"time_min": "2026-11-23T00:00:00-08:00", "time_max": "2026-11-24T00:00:00-08:00"},
                )
                assert not ev_list.is_error
                assert list_events_attempts == 2  # 429 then 200
                assert json.loads(ev_list.content[0].text)["count"] == 1

                # Create / update / delete (external -> trigger elicitation)
                c_res = await session.call_tool(
                    "gcal_create_event",
                    {
                        "summary": "Dinner",
                        "start": "2026-11-23T19:00:00-08:00",
                        "end": "2026-11-23T20:00:00-08:00",
                    },
                )
                assert not c_res.is_error
                u_res = await session.call_tool("gcal_update_event", {"event_id": "ev-2", "summary": "Dinner Updated"})
                assert not u_res.is_error
                d_res = await session.call_tool("gcal_delete_event", {"event_id": "ev-2"})
                assert not d_res.is_error

                # 4. Gmail search (concurrency <= 5), read (plain + HTML strip), send (MIME threading + elicitation)
                g_search = await session.call_tool("gmail_search", {"query": "hello", "limit": 8})
                assert not g_search.is_error
                assert json.loads(g_search.content[0].text)["count"] == 8
                assert 1 <= max_concurrent_metadata <= 5

                g_plain = json.loads((await session.call_tool("gmail_read", {"message_id": "m0"})).content[0].text)
                assert g_plain["body"] == "Plain body text"

                g_html = json.loads((await session.call_tool("gmail_read", {"message_id": "m-html"})).content[0].text)
                assert "Hello" in g_html["body"] and "World" in g_html["body"]
                assert "evil()" not in g_html["body"] and "color:red" not in g_html["body"]

                elicited_before_send = len(elicited_tools)
                g_send = await session.call_tool(
                    "gmail_send",
                    {
                        "to": "zubin.test@example.com",
                        "subject": "Re: Hello m0",
                        "body": "Sounds good!",
                        "reply_to_message_id": "m0",
                    },
                )
                assert not g_send.is_error
                assert len(elicited_tools) == elicited_before_send + 1
                assert "gmail_send" in elicited_tools[-1]

                assert len(sent_payloads) == 1
                assert sent_payloads[0]["threadId"] == "thread-m0"
                raw_mime = base64.urlsafe_b64decode(sent_payloads[0]["raw"].encode("ascii"))
                parsed_mime = email.message_from_bytes(raw_mime)
                assert parsed_mime["To"] == "zubin.test@example.com"
                assert parsed_mime["In-Reply-To"] == "<m0@example.com>"
                assert parsed_mime["References"] == "<root@example.com> <m0@example.com>"

                # 5. Token refresh on 401 succeeds transparently
                g_ref = await session.call_tool("gmail_read", {"message_id": "m-401-then-ok"})
                assert not g_ref.is_error
                assert refresh_calls == 1
                assert json.loads(g_ref.content[0].text)["body"] == "Worked after refresh"

                # 6. Revoked refresh token (invalid_grant) -> takes BOTH gcal and gmail offline + emits list_changed
                revoke_next_refresh = True
                list_changed_ev = anyio.Event()
                g_rev = await session.call_tool("gmail_read", {"message_id": "m-revoke"})
                assert g_rev.is_error
                with anyio.fail_after(5):
                    await list_changed_ev.wait()

                tools_after_revoke = {t.name for t in (await session.list_tools()).tools}
                assert not any(n.startswith(("gcal_", "gmail_")) for n in tools_after_revoke)
