"""Unit & integration tests for typedstream decoding and chat.db FTS5 MessagesIndex."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from zos_actuator.messages import APPLE_EPOCH, MessagesIndex, apple_timestamp_to_iso
from zos_actuator.typedstream import build_typedstream_attributed_body, decode_attributed_body


def _apple_ns(dt: datetime) -> int:
    return int((dt - APPLE_EPOCH).total_seconds() * 1_000_000_000)


def _create_fixture_chat_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE handle (
            ROWID INTEGER PRIMARY KEY,
            id TEXT
        );
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY,
            chat_identifier TEXT
        );
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            attributedBody BLOB,
            is_from_me INTEGER,
            date INTEGER,
            handle_id INTEGER
        );
        CREATE TABLE chat_message_join (
            chat_id INTEGER,
            message_id INTEGER
        );
        """
    )
    return conn


def test_typedstream_decode_short_long_utf8_and_invalid() -> None:
    short = "Hello ZOS!"
    assert decode_attributed_body(build_typedstream_attributed_body(short)) == short

    # > 127 bytes triggers 0x81 2-byte length prefix + multi-byte UTF-8 emoji
    long_msg = "Lunch at 12:30 ☕️ — " + ("architecture notes " * 15)
    assert len(long_msg.encode("utf-8")) > 200
    assert decode_attributed_body(build_typedstream_attributed_body(long_msg)) == long_msg

    assert decode_attributed_body(None) == ""
    assert decode_attributed_body(b"") == ""
    assert decode_attributed_body(b"not a typedstream blob") == ""


def test_apple_timestamp_to_iso() -> None:
    dt = datetime(2026, 11, 23, 18, 30, 0, tzinfo=UTC)
    assert apple_timestamp_to_iso(_apple_ns(dt)) == "2026-11-23T18:30:00Z"
    # Legacy seconds format
    sec = int((dt - APPLE_EPOCH).total_seconds())
    assert apple_timestamp_to_iso(sec) == "2026-11-23T18:30:00Z"


def test_messages_index_sync_attributed_body_and_filters(tmp_path: Path) -> None:
    chat_db = tmp_path / "chat.db"
    idx_db = tmp_path / "messages.db"
    conn = _create_fixture_chat_db(chat_db)

    conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+14155550101'), (2, 'alice@example.com')")
    conn.execute("INSERT INTO chat (ROWID, chat_identifier) VALUES (1, '+14155550101'), (2, 'alice@example.com')")

    t1 = _apple_ns(datetime(2026, 11, 20, 10, 0, 0, tzinfo=UTC))
    t2 = _apple_ns(datetime(2026, 11, 22, 15, 30, 0, tzinfo=UTC))
    t3 = _apple_ns(datetime(2026, 11, 23, 9, 0, 0, tzinfo=UTC))

    # Row 1: plain text
    conn.execute(
        "INSERT INTO message (ROWID, text, attributedBody, is_from_me, date, handle_id) VALUES (?, ?, ?, ?, ?, ?)",
        (1, "Let's grab espresso tomorrow morning", None, 0, t1, 1),
    )
    conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 1)")

    # Row 2: NULL text, content only in attributedBody (macOS Ventura+ behavior)
    blob = build_typedstream_attributed_body("Flight confirmation for Tokyo espresso trip")
    conn.execute(
        "INSERT INTO message (ROWID, text, attributedBody, is_from_me, date, handle_id) VALUES (?, ?, ?, ?, ?, ?)",
        (2, None, blob, 1, t2, 2),
    )
    conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (2, 2)")
    conn.commit()

    index = MessagesIndex(chat_db, idx_db)

    # First search syncs 2 rows and finds both
    res1 = index.search("espresso")
    assert res1["synced_new"] == 2
    assert res1["count"] == 2
    assert res1["matches"][0]["handle"] == "alice@example.com"
    assert res1["matches"][0]["is_from_me"] is True
    assert res1["matches"][0]["date"] == "2026-11-22T15:30:00Z"
    assert "Tokyo" in res1["matches"][0]["text"]

    # Second search syncs 0 new rows
    res2 = index.search("espresso", handle="alice")
    assert res2["synced_new"] == 0
    assert res2["count"] == 1

    # since filter
    res_since = index.search("espresso", since="2026-11-21T00:00:00Z")
    assert res_since["count"] == 1
    assert res_since["matches"][0]["date"] == "2026-11-22T15:30:00Z"

    # Append a 3rd message -> incremental ROWID sync picks up only ROWID 3
    conn.execute(
        "INSERT INTO message (ROWID, text, attributedBody, is_from_me, date, handle_id) VALUES (?, ?, ?, ?, ?, ?)",
        (3, "New espresso roast arrived", None, 0, t3, 1),
    )
    conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 3)")
    conn.commit()
    conn.close()

    res3 = index.search("espresso", limit=2)
    assert res3["synced_new"] == 1
    assert res3["count"] == 2
    assert "roast" in res3["matches"][0]["text"]

    # Verify FTS5 external-content index integrity after incremental syncs
    with sqlite3.connect(idx_db) as idx_check:
        idx_check.execute("INSERT INTO messages_fts(messages_fts) VALUES('integrity-check')")
