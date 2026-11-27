"""Synthetic Messages/AddressBook fixtures. Never touches real personal data."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from zos_mac.contacts import ContactsIndex
from zos_mac.typedstream import build_typedstream_attributed_body

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
BASE = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


def apple_ns(minutes: int) -> int:
    dt = BASE + timedelta(minutes=minutes)
    return int((dt - APPLE_EPOCH).total_seconds() * 1_000_000_000)


JARED_ID = "AB-JARED:ABPerson"

# (pk, unique_id, first, last, org, nick, phones[(label, number)], emails[(label, addr)])
CONTACTS: list[tuple] = [
    (
        1,
        JARED_ID,
        "Jared",
        "Dunn",
        None,
        "JD",
        [("_$!<Mobile>!$_", "+1 (415) 555-0101"), ("_$!<Work>!$_", "650-555-0102")],
        [("_$!<Home>!$_", "Jared@Example.com")],
    ),
    (2, None, "Monica", "Hall", None, None, [("_$!<Mobile>!$_", "+14155550103")], []),
    (
        3,
        None,
        "Gavin",
        "Belson",
        "Hooli",
        None,
        [("_$!<Mobile>!$_", "+14155550104"), ("iPhone", "+14155550105")],
        [],
    ),
    (4, None, None, None, "Pied Piper", None, [], [("_$!<Work>!$_", "team@piedpiper.example")]),
    (
        5,
        None,
        "Richard",
        "Hendricks",
        None,
        "Richie",
        [("_$!<Mobile>!$_", "+14155550106")],
        [("_$!<Home>!$_", "richard@example.com")],
    ),
]


def create_addressbook(root: Path, contacts: Sequence[tuple] = CONTACTS) -> Path:
    db = root / "Sources" / "SRC-A" / "AddressBook-v22.abcddb"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY, ZUNIQUEID TEXT, ZFIRSTNAME TEXT, ZLASTNAME TEXT,
                ZORGANIZATION TEXT, ZNICKNAME TEXT
            );
            CREATE TABLE ZABCDPHONENUMBER (
                Z_PK INTEGER PRIMARY KEY AUTOINCREMENT, ZOWNER INTEGER, ZFULLNUMBER TEXT,
                ZLABEL TEXT, ZORDERINGINDEX INTEGER
            );
            CREATE TABLE ZABCDEMAILADDRESS (
                Z_PK INTEGER PRIMARY KEY AUTOINCREMENT, ZOWNER INTEGER, ZADDRESS TEXT,
                ZADDRESSNORMALIZED TEXT, ZLABEL TEXT, ZORDERINGINDEX INTEGER
            );
            """
        )
        for pk, uid, first, last, org, nick, phones, emails in contacts:
            conn.execute(
                "INSERT INTO ZABCDRECORD VALUES (?, ?, ?, ?, ?, ?)", (pk, uid, first, last, org, nick)
            )
            for i, (label, number) in enumerate(phones):
                conn.execute(
                    "INSERT INTO ZABCDPHONENUMBER (ZOWNER, ZFULLNUMBER, ZLABEL, ZORDERINGINDEX) VALUES (?, ?, ?, ?)",
                    (pk, number, label, i),
                )
            for i, (label, addr) in enumerate(emails):
                conn.execute(
                    "INSERT INTO ZABCDEMAILADDRESS (ZOWNER, ZADDRESS, ZADDRESSNORMALIZED, ZLABEL, ZORDERINGINDEX) VALUES (?, ?, ?, ?, ?)",
                    (pk, addr, addr.lower(), label, i),
                )
        conn.commit()
    finally:
        conn.close()
    return db


HANDLES = {
    1: "+14155550101",  # Jared mobile
    2: "+16505550102",  # Jared work
    3: "+14155550103",  # Monica
    4: "+19995550000",  # unknown
    5: "+14155550106",  # Richard phone
    6: "richard@example.com",  # Richard email
}

# chat rowid -> (guid, identifier, display_name, style, service, handle rowids)
CHATS = {
    1: ("iMessage;-;+14155550101", "+14155550101", None, 45, "iMessage", [1]),
    2: ("SMS;-;+16505550102", "+16505550102", None, 45, "SMS", [2]),
    3: ("iMessage;+;chat1001", "chat1001", "Hooli Survivors", 43, "iMessage", [1, 3]),
    4: ("iMessage;+;chat1002", "chat1002", "", 43, "iMessage", [3, 4]),
    5: ("iMessage;-;+14155550106", "+14155550106", None, 45, "iMessage", [5]),
    6: ("iMessage;-;richard@example.com", "richard@example.com", None, 45, "iMessage", [6]),
    7: ("iMessage;-;+14155550103", "+14155550103", None, 45, "iMessage", [3]),
}

# (rowid, chat, minutes, from_me, handle_rowid, text, is_read, attributed_only, assoc_type)
MESSAGES = [
    (1, 1, 0, 0, 1, "Jared one", 1, False, 0),
    (2, 1, 1, 1, 1, "me two", 1, False, 0),
    (3, 1, 2, 0, 1, "Jared three (attributed)", 1, True, 0),
    (4, 1, 3, 1, 1, "me four", 1, False, 0),
    (5, 1, 4, 0, 1, "Jared five unread", 0, False, 0),
    (6, 1, 4, 0, 1, "Loved \u201cme four\u201d", 0, False, 2001),  # tapback, skipped
    (7, 2, -60, 0, 2, "Jared from work phone", 1, False, 0),
    (8, 3, 10, 0, 3, "Monica in the group", 0, False, 0),
    (9, 3, 11, 1, 0, "me in the group", 1, False, 0),
    (10, 4, 5, 0, 4, "unknown in unnamed group", 1, False, 0),
    (11, 5, -120, 0, 5, "Richard by phone", 1, False, 0),
    (12, 6, -30, 1, 6, "Richard by email", 1, False, 0),
    (13, 7, -10, 0, 3, "Monica direct", 1, False, 0),
]


def create_chat_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT, service TEXT);
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY, guid TEXT, chat_identifier TEXT, display_name TEXT,
                style INTEGER, service_name TEXT
            );
            CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, attributedBody BLOB,
                is_from_me INTEGER, is_read INTEGER DEFAULT 0, date INTEGER, handle_id INTEGER,
                service TEXT, cache_has_attachments INTEGER DEFAULT 0,
                associated_message_type INTEGER DEFAULT 0, error INTEGER DEFAULT 0
            );
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
            """
        )
        for rowid, hid in HANDLES.items():
            conn.execute("INSERT INTO handle VALUES (?, ?, 'iMessage')", (rowid, hid))
        for rowid, (guid, ident, name, style, svc, handles) in CHATS.items():
            conn.execute("INSERT INTO chat VALUES (?, ?, ?, ?, ?, ?)", (rowid, guid, ident, name, style, svc))
            for h in handles:
                conn.execute("INSERT INTO chat_handle_join VALUES (?, ?)", (rowid, h))
        for rowid, chat, mins, from_me, hid, text, is_read, attributed, assoc in MESSAGES:
            conn.execute(
                "INSERT INTO message (ROWID, text, attributedBody, is_from_me, is_read, date, handle_id, service, associated_message_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rowid,
                    None if attributed else text,
                    build_typedstream_attributed_body(text) if attributed else None,
                    from_me,
                    is_read,
                    apple_ns(mins),
                    hid,
                    CHATS[chat][4],
                    assoc,
                ),
            )
            conn.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat, rowid))
        conn.commit()
    finally:
        conn.close()
    return path


def insert_outgoing(chat_db: Path, chat_rowid: int, text: str, *, error: int = 0) -> int:
    """Simulate Messages.app recording our outgoing message in the fixture chat.db."""
    conn = sqlite3.connect(chat_db)
    try:
        hid = CHATS[chat_rowid][5][0]
        cur = conn.execute(
            "INSERT INTO message (text, is_from_me, is_read, date, handle_id, service, error) VALUES (?, 1, 1, ?, ?, 'iMessage', ?)",
            (text, apple_ns(1000), hid, error),
        )
        rowid = int(cur.lastrowid or 0)
        conn.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat_rowid, rowid))
        conn.commit()
        return rowid
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _never_real_addressbook(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Privacy guard: a ContactsIndex built without an explicit root must not read the real one."""
    import zos_mac.contacts as contacts_mod

    empty = tmp_path_factory.mktemp("no-addressbook") / "AddressBook"
    monkeypatch.setattr(contacts_mod, "DEFAULT_ADDRESSBOOK_ROOT", empty)


@dataclass
class MacFixture:
    root: Path
    addressbook_root: Path
    chat_db: Path
    index_db: Path
    contacts: ContactsIndex


@pytest.fixture
def mac_fixture(tmp_path: Path) -> MacFixture:
    ab_root = tmp_path / "AddressBook"
    create_addressbook(ab_root)
    chat_db = create_chat_db(tmp_path / "chat.db")
    return MacFixture(
        root=tmp_path,
        addressbook_root=ab_root,
        chat_db=chat_db,
        index_db=tmp_path / "messages.db",
        contacts=ContactsIndex(ab_root),
    )
