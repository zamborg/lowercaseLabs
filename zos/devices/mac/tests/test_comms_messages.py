"""Messages threads, read pagination, sender resolution, and search chat_id (fixtures only)."""

from __future__ import annotations

import pytest

from conftest import JARED_ID, MacFixture
from zos_mac.chatdb import ChatStore
from zos_mac.messages import MessagesIndex


def _store(fx: MacFixture) -> ChatStore:
    return ChatStore(fx.chat_db, fx.contacts)


def test_threads_order_group_names_participants_unread(mac_fixture: MacFixture) -> None:
    res = _store(mac_fixture).threads(limit=20)
    ids = [t["chat_id"] for t in res["threads"]]
    assert ids == [
        "iMessage;+;chat1001",
        "iMessage;+;chat1002",
        "iMessage;-;+14155550101",
        "iMessage;-;+14155550103",
        "iMessage;-;richard@example.com",
        "SMS;-;+16505550102",
        "iMessage;-;+14155550106",
    ]
    by_id = {t["chat_id"]: t for t in res["threads"]}

    named = by_id["iMessage;+;chat1001"]
    assert named["display_name"] == "Hooli Survivors"
    assert named["is_group"] is True
    assert named["participants"] == [
        {"handle": "+14155550101", "name": "Jared Dunn"},
        {"handle": "+14155550103", "name": "Monica Hall"},
    ]
    assert named["last_message"] == {"from_me": True, "snippet": "me in the group"}
    assert named["unread_count"] == 1

    unnamed = by_id["iMessage;+;chat1002"]
    assert unnamed["display_name"] == "Monica Hall, +19995550000"
    assert unnamed["participants"][1] == {"handle": "+19995550000", "name": None}

    jared = by_id["iMessage;-;+14155550101"]
    assert jared["display_name"] == "Jared Dunn"
    assert jared["is_group"] is False
    # Tapback (associated_message_type 2001) is neither the snippet nor unread
    assert jared["last_message"]["snippet"] == "Jared five unread"
    assert jared["unread_count"] == 1
    assert jared["last_message_at"] == "2026-11-20T12:04:00Z"


def test_threads_limit_since_and_query(mac_fixture: MacFixture) -> None:
    store = _store(mac_fixture)
    assert store.threads(limit=2)["count"] == 2
    since = store.threads(since="2026-11-20T12:00:00Z")
    assert [t["chat_id"] for t in since["threads"]] == [
        "iMessage;+;chat1001",
        "iMessage;+;chat1002",
        "iMessage;-;+14155550101",
    ]
    monica = store.threads(query="monica")
    assert {t["chat_id"] for t in monica["threads"]} == {
        "iMessage;+;chat1001",
        "iMessage;+;chat1002",
        "iMessage;-;+14155550103",
    }
    assert [t["chat_id"] for t in store.threads(query="hooli surv")["threads"]] == ["iMessage;+;chat1001"]
    by_phone = store.threads(query="(650) 555-0102")
    assert [t["chat_id"] for t in by_phone["threads"]] == ["SMS;-;+16505550102"]


def test_read_pagination_chronological_and_sender_resolution(mac_fixture: MacFixture) -> None:
    store = _store(mac_fixture)
    page1 = store.read("iMessage;-;+14155550101", limit=2)
    assert [m["text"] for m in page1["messages"]] == ["me four", "Jared five unread"]
    assert page1["has_more"] is True
    assert page1["messages"][0]["sender"] == "me"
    assert page1["messages"][0]["handle"] is None
    assert page1["messages"][1]["sender"] == "Jared Dunn"
    assert page1["messages"][1]["handle"] == "+14155550101"
    assert page1["chat"]["chat_id"] == "iMessage;-;+14155550101"

    page2 = store.read("iMessage;-;+14155550101", limit=2, before=page1["next_before"])
    assert [m["text"] for m in page2["messages"]] == ["me two", "Jared three (attributed)"]
    assert page2["has_more"] is True

    page3 = store.read("iMessage;-;+14155550101", limit=2, before=page2["next_before"])
    assert [m["text"] for m in page3["messages"]] == ["Jared one"]
    assert page3["has_more"] is False
    assert page3["next_before"] is None

    # ISO `before` works too
    iso = store.read("iMessage;-;+14155550101", limit=10, before="2026-11-20T12:02:00Z")
    assert [m["text"] for m in iso["messages"]] == ["Jared one", "me two"]


def test_read_by_handle_contact_id_and_group(mac_fixture: MacFixture) -> None:
    store = _store(mac_fixture)
    by_handle = store.read("(415) 555-0101")
    assert by_handle["chat"]["chat_id"] == "iMessage;-;+14155550101"
    assert "merged_chat_ids" not in by_handle

    merged = store.read(JARED_ID)
    assert merged["merged_chat_ids"] == ["iMessage;-;+14155550101", "SMS;-;+16505550102"]
    assert merged["messages"][0]["text"] == "Jared from work phone"
    assert merged["count"] == 6

    group = store.read("chat1001")
    assert group["chat"]["display_name"] == "Hooli Survivors"
    assert [(m["sender"], m["text"]) for m in group["messages"]] == [
        ("Monica Hall", "Monica in the group"),
        ("me", "me in the group"),
    ]

    with pytest.raises(LookupError):
        store.read("Jared")  # names are not chat refs; use contacts_search first
    with pytest.raises(LookupError):
        store.read("+15555550000")


def test_messages_search_results_include_chat_id(mac_fixture: MacFixture) -> None:
    idx = MessagesIndex(mac_fixture.chat_db, mac_fixture.index_db, contacts=mac_fixture.contacts)
    res = idx.search("group")
    by_text = {m["text"]: m for m in res["matches"]}
    assert by_text["me in the group"]["chat_id"] == "iMessage;+;chat1001"
    assert by_text["unknown in unnamed group"]["chat_id"] == "iMessage;+;chat1002"
    # The chat_id chains straight into messages_read
    read = ChatStore(mac_fixture.chat_db, mac_fixture.contacts).read(by_text["me in the group"]["chat_id"])
    assert read["chat"]["display_name"] == "Hooli Survivors"
