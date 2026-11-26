"""M8.5 tests for AddressBook contact lookup, mtime invalidation, name->handle resolution, and graceful degradation."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from zos_mac.contacts import ContactsIndex, normalize_email, normalize_phone
from zos_mac.messages import MessagesIndex


def _create_fixture_abcddb(path: Path, contacts: list[tuple[int, str | None, str | None, str | None, list[str], list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE ZABCDRECORD (
                Z_PK INTEGER PRIMARY KEY,
                ZFIRSTNAME TEXT,
                ZLASTNAME TEXT,
                ZORGANIZATION TEXT,
                ZNICKNAME TEXT
            );
            CREATE TABLE ZABCDPHONENUMBER (
                Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
                ZOWNER INTEGER,
                ZFULLNUMBER TEXT
            );
            CREATE TABLE ZABCDEMAILADDRESS (
                Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
                ZOWNER INTEGER,
                ZADDRESS TEXT,
                ZADDRESSNORMALIZED TEXT
            );
            """
        )
        for pk, first, last, org, phones, emails in contacts:
            conn.execute(
                "INSERT INTO ZABCDRECORD (Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION) VALUES (?, ?, ?, ?)",
                (pk, first, last, org),
            )
            for p in phones:
                conn.execute("INSERT INTO ZABCDPHONENUMBER (ZOWNER, ZFULLNUMBER) VALUES (?, ?)", (pk, p))
            for e in emails:
                conn.execute(
                    "INSERT INTO ZABCDEMAILADDRESS (ZOWNER, ZADDRESS, ZADDRESSNORMALIZED) VALUES (?, ?, ?)",
                    (pk, e, e.lower()),
                )
        conn.commit()
    finally:
        conn.close()


def _create_fixture_chat_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
            CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT);
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                date INTEGER,
                handle_id INTEGER
            );
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);

            INSERT INTO handle (ROWID, id) VALUES
                (1, '+14155550199'),
                (2, 'Alice.Smith@Example.COM'),
                (3, '+19995550000');
            INSERT INTO chat (ROWID, chat_identifier) VALUES
                (1, '+14155550199'),
                (2, 'Alice.Smith@Example.COM'),
                (3, '+19995550000');
            INSERT INTO message (ROWID, text, is_from_me, date, handle_id) VALUES
                (1, 'Dinner plans on Friday', 0, 750000000000000000, 1),
                (2, 'Dinner reservation confirmed by email', 0, 750000060000000000, 2),
                (3, 'Dinner from unknown caller', 0, 750000120000000000, 3);
            INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 1), (2, 2), (3, 3);
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_contacts_normalization_mtime_cache_and_messages_name_search(tmp_path: Path) -> None:
    assert normalize_phone("+1 (415) 555-0199") == "4155550199"
    assert normalize_phone("415-555-0199") == "4155550199"
    assert normalize_email("  Alice@Example.COM ") == "alice@example.com"

    ab_root = tmp_path / "AddressBook"
    src_db = ab_root / "Sources" / "SRC-1" / "AddressBook-v22.abcddb"
    top_db = ab_root / "AddressBook-v22.abcddb"

    _create_fixture_abcddb(
        src_db,
        [
            (1, "Alice", "Smith", None, ["(415) 555-0199"], ["alice.smith@example.com"]),
        ],
    )
    _create_fixture_abcddb(
        top_db,
        [
            (2, None, None, "Acme Corp", ["+1-650-555-0123"], []),
        ],
    )

    contacts = ContactsIndex(ab_root)
    assert contacts.lookup_name("+14155550199") == "Alice Smith"
    assert contacts.lookup_name("ALICE.SMITH@EXAMPLE.COM") == "Alice Smith"
    assert contacts.lookup_name("6505550123") == "Acme Corp"
    assert contacts.lookup_name("+19995550000") is None
    assert contacts.last_hint is None

    # Mtime invalidation: update src_db with a new contact and bump mtime
    _create_fixture_abcddb(
        src_db,
        [
            (1, "Alice", "Smith", None, ["(415) 555-0199"], ["alice.smith@example.com"]),
            (3, "Bob", "Jones", None, ["+1 (212) 555-0144"], []),
        ],
    )
    st = src_db.stat()
    os.utime(src_db, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert contacts.lookup_name("+12125550144") == "Bob Jones"

    # MessagesIndex integration: contact_name populated (null if unknown) + handle accepts contact name
    chat_db = tmp_path / "chat.db"
    index_db = tmp_path / "index.db"
    _create_fixture_chat_db(chat_db)

    idx = MessagesIndex(chat_db, index_db, contacts=contacts)
    all_res = idx.search("Dinner")
    assert all_res["count"] == 3
    assert "contacts_hint" not in all_res
    by_handle = {m["handle"]: m["contact_name"] for m in all_res["matches"]}
    assert by_handle == {
        "+14155550199": "Alice Smith",
        "Alice.Smith@Example.COM": "Alice Smith",
        "+19995550000": None,
    }

    # Filter by contact name ("Alice") -> resolves to BOTH her phone and email handles!
    alice_res = idx.search("Dinner", handle="Alice")
    assert alice_res["count"] == 2
    assert {m["handle"] for m in alice_res["matches"]} == {"+14155550199", "Alice.Smith@Example.COM"}
    assert all(m["contact_name"] == "Alice Smith" for m in alice_res["matches"])


def test_contacts_unreadable_degrades_gracefully_with_hint(tmp_path: Path) -> None:
    ab_root = tmp_path / "AddressBook"
    src_db = ab_root / "Sources" / "SRC-1" / "AddressBook-v22.abcddb"
    _create_fixture_abcddb(src_db, [(1, "Alice", "Smith", None, ["+14155550199"], [])])

    chat_db = tmp_path / "chat.db"
    index_db = tmp_path / "index.db"
    _create_fixture_chat_db(chat_db)

    src_db.chmod(0o000)
    try:
        idx = MessagesIndex(chat_db, index_db, contacts=ContactsIndex(ab_root))
        res = idx.search("Dinner")
        assert res["count"] == 3
        assert all(m["contact_name"] is None for m in res["matches"])
        assert "contacts_hint" in res
        assert "Full Disk Access" in res["contacts_hint"]
    finally:
        src_db.chmod(0o600)
