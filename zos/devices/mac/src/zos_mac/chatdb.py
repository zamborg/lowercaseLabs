"""Read-only views over macOS Messages chat.db: threads, conversations, and send resolution.

Never logs message text, names, or handles.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from zos_mac.contacts import (
    Contact,
    ContactsIndex,
    looks_like_phone,
    normalize_email,
    normalize_handle,
    normalize_phone,
)
from zos_mac.typedstream import decode_attributed_body

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
MAX_THREADS_LIMIT = 100
MAX_READ_LIMIT = 200
MAX_TEXT_CHARS = 2000
SNIPPET_CHARS = 120
GROUP_CHAT_STYLE = 43


def apple_timestamp_to_iso(raw_date: int | float | None) -> str:
    """Convert Apple Messages date (ns or s since 2001-01-01 UTC) to ISO 8601 UTC."""
    if raw_date is None or raw_date == 0:
        return APPLE_EPOCH.isoformat().replace("+00:00", "Z")
    val = float(raw_date)
    seconds = val / 1_000_000_000.0 if abs(val) >= 1e12 else val
    dt = APPLE_EPOCH + timedelta(seconds=seconds)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_to_apple_ns(value: str) -> int:
    """Parse an ISO 8601 timestamp (naive = UTC) into Apple-epoch nanoseconds."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int((dt - APPLE_EPOCH).total_seconds() * 1_000_000_000)


def _date_ns(raw: int | float | None) -> int:
    """Normalize a chat.db date to nanoseconds (legacy rows store seconds)."""
    if not raw:
        return 0
    val = int(raw)
    return val if abs(val) >= 10**12 else val * 1_000_000_000


def open_chat_db_ro(chat_db_path: Path) -> sqlite3.Connection:
    """Open chat.db strictly read-only, surfacing macOS Full Disk Access guidance on failure."""
    if not chat_db_path.exists():
        raise FileNotFoundError(f"Messages database not found at {chat_db_path}")
    uri = f"file:{quote(str(chat_db_path), safe='/')}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 2000")
        # Probe readability immediately so macOS TCC/FDA errors surface clearly
        conn.execute("SELECT 1 FROM message LIMIT 1")
        return conn
    except (PermissionError, sqlite3.OperationalError) as exc:
        resolved_py = os.path.realpath(sys.executable)
        raise PermissionError(
            f"Cannot read {chat_db_path} ({exc}). "
            f"Grant Full Disk Access in System Settings -> Privacy & Security -> Full Disk Access to: "
            f"{resolved_py}"
        ) from exc


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) > limit:
        return text[:limit] + "...", True
    return text, False


class AmbiguousRecipientError(ValueError):
    """Raised when a contact has several handles and none is clearly the one to use."""

    def __init__(self, message: str, options: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.options = options


@dataclass(slots=True)
class _Chat:
    rowid: int
    guid: str
    identifier: str
    display_name: str | None
    style: int | None
    service: str | None
    last_date: int
    handles: list[str] = field(default_factory=list)

    @property
    def chat_id(self) -> str:
        return self.guid or self.identifier

    @property
    def is_group(self) -> bool:
        if self.style is not None:
            return self.style == GROUP_CHAT_STYLE
        return len(self.handles) > 1


@dataclass(slots=True)
class SendTarget:
    kind: str  # "chat" | "handle"
    value: str  # chat guid or handle
    chat_id: str | None = None
    contact_name: str | None = None


class ChatStore:
    """Read-only queries over chat.db, resolving handles to contact names."""

    def __init__(self, chat_db_path: Path, contacts: ContactsIndex) -> None:
        self.chat_db_path = chat_db_path.expanduser()
        self.contacts = contacts

    # ---- low-level helpers -------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        return open_chat_db_ro(self.chat_db_path)

    @staticmethod
    def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> set[str]:
        return {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    def _load_chats(self, conn: sqlite3.Connection) -> list[_Chat]:
        cols = self._cols(conn, "chat")

        def sel(name: str) -> str:
            return f"c.{name}" if name in cols else f"NULL AS {name}"

        rows = conn.execute(
            f"""
            SELECT c.ROWID AS rowid, {sel('guid')}, {sel('chat_identifier')},
                   {sel('display_name')}, {sel('style')}, {sel('service_name')},
                   (SELECT MAX(m.date) FROM chat_message_join cmj
                      JOIN message m ON m.ROWID = cmj.message_id
                     WHERE cmj.chat_id = c.ROWID) AS last_date
            FROM chat c
            """
        ).fetchall()
        chats = {
            int(r["rowid"]): _Chat(
                rowid=int(r["rowid"]),
                guid=str(r["guid"] or ""),
                identifier=str(r["chat_identifier"] or ""),
                display_name=(str(r["display_name"]).strip() or None) if r["display_name"] else None,
                style=int(r["style"]) if r["style"] is not None else None,
                service=str(r["service_name"]) if r["service_name"] else None,
                last_date=int(r["last_date"] or 0),
            )
            for r in rows
        }
        if "chat_handle_join" in self._tables(conn):
            for r in conn.execute(
                """
                SELECT chj.chat_id AS chat_id, h.id AS handle
                FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id
                ORDER BY chj.chat_id, h.ROWID
                """
            ):
                chat = chats.get(int(r["chat_id"]))
                if chat is not None and r["handle"] and r["handle"] not in chat.handles:
                    chat.handles.append(str(r["handle"]))
        for chat in chats.values():
            if not chat.handles and chat.identifier and normalize_handle(chat.identifier):
                chat.handles.append(chat.identifier)
        return list(chats.values())

    def _name(self, handle: str | None) -> str | None:
        return self.contacts.lookup_name(handle) if handle else None

    def _participants(self, chat: _Chat) -> list[dict[str, Any]]:
        return [{"handle": h, "name": self._name(h)} for h in chat.handles]

    def _display_name(self, chat: _Chat) -> str:
        if chat.display_name:
            return chat.display_name
        names = [self._name(h) or h for h in chat.handles]
        if names:
            return ", ".join(names)
        return chat.identifier or chat.guid

    def _chat_summary(self, chat: _Chat) -> dict[str, Any]:
        return {
            "chat_id": chat.chat_id,
            "display_name": self._display_name(chat),
            "is_group": chat.is_group,
            "service": chat.service,
            "participants": self._participants(chat),
        }

    @staticmethod
    def _message_text(row: sqlite3.Row) -> str:
        raw = row["text"]
        text = raw if raw else decode_attributed_body(row["attributed_body"])
        if text:
            return text.replace("\ufffc", "").strip() or "[attachment]"
        if row["has_attachments"]:
            return "[attachment]"
        return ""

    def _message_select(self, conn: sqlite3.Connection) -> tuple[str, str]:
        """Return (SELECT column list, extra WHERE filter) tolerant of older schemas."""
        cols = self._cols(conn, "message")

        def sel(name: str, alias: str) -> str:
            return f"m.{name} AS {alias}" if name in cols else f"NULL AS {alias}"

        select = ", ".join(
            [
                "m.ROWID AS rowid",
                "m.date AS date_raw",
                "m.is_from_me AS is_from_me",
                "m.text AS text",
                sel("attributedBody", "attributed_body"),
                sel("cache_has_attachments", "has_attachments"),
                sel("service", "service"),
                sel("error", "error"),
                "h.id AS handle",
            ]
        )
        # Skip tapbacks/reactions (they are not conversational messages)
        extra = " AND COALESCE(m.associated_message_type, 0) = 0" if "associated_message_type" in cols else ""
        return select, extra

    # ---- threads -----------------------------------------------------------

    def _chat_matches(self, chat: _Chat, query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        norm = normalize_handle(q) if ("@" in q or looks_like_phone(q)) else None
        if norm and any(normalize_handle(h) == norm for h in chat.handles):
            return True
        haystack = [self._display_name(chat), chat.identifier]
        for h in chat.handles:
            haystack.append(h)
            name = self._name(h)
            if name:
                haystack.append(name)
        return any(q in s.lower() for s in haystack if s)

    def threads(
        self,
        *,
        limit: int = 20,
        since: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Recent conversations, newest first."""
        clamped = max(1, min(int(limit), MAX_THREADS_LIMIT))
        since_ns = iso_to_apple_ns(since) if since and since.strip() else None
        conn = self._open()
        try:
            chats = [c for c in self._load_chats(conn) if c.last_date]
            chats.sort(key=lambda c: (_date_ns(c.last_date), c.rowid), reverse=True)
            selected: list[_Chat] = []
            for chat in chats:
                if since_ns is not None and _date_ns(chat.last_date) < since_ns:
                    break
                if query and not self._chat_matches(chat, query):
                    continue
                selected.append(chat)
                if len(selected) >= clamped:
                    break

            select, extra = self._message_select(conn)
            msg_cols = self._cols(conn, "message")
            out_threads: list[dict[str, Any]] = []
            for chat in selected:
                last = conn.execute(
                    f"""
                    SELECT {select}
                    FROM message m
                    JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                    LEFT JOIN handle h ON h.ROWID = m.handle_id
                    WHERE cmj.chat_id = ?{extra}
                    ORDER BY m.date DESC, m.ROWID DESC
                    LIMIT 1
                    """,
                    (chat.rowid,),
                ).fetchone()
                unread: int | None = None
                if "is_read" in msg_cols:
                    unread = int(
                        conn.execute(
                            f"""
                            SELECT COUNT(*) FROM message m
                            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                            WHERE cmj.chat_id = ? AND m.is_from_me = 0
                              AND COALESCE(m.is_read, 0) = 0{extra}
                            """,
                            (chat.rowid,),
                        ).fetchone()[0]
                    )
                summary = self._chat_summary(chat)
                summary["last_message_at"] = apple_timestamp_to_iso(chat.last_date)
                if last is not None:
                    snippet, _ = _truncate(self._message_text(last), SNIPPET_CHARS)
                    summary["last_message"] = {"from_me": bool(last["is_from_me"]), "snippet": snippet}
                summary["unread_count"] = unread
                out_threads.append(summary)
        finally:
            conn.close()

        out: dict[str, Any] = {"count": len(out_threads), "threads": out_threads}
        if self.contacts.last_hint:
            out["contacts_hint"] = self.contacts.last_hint
        return out

    # ---- chat resolution ---------------------------------------------------

    def _handles_for_contact(self, contact: Contact) -> set[str]:
        return {h.normalized for h in contact.handles}

    def _resolve_chats(self, conn: sqlite3.Connection, ref: str) -> tuple[list[_Chat], str | None]:
        """Map a chat id, handle, or contact id to matching chats (most recent first)."""
        r = ref.strip()
        if not r:
            raise ValueError("chat must be non-empty")
        chats = self._load_chats(conn)
        by_recent = sorted(chats, key=lambda c: (_date_ns(c.last_date), c.rowid), reverse=True)

        exact = [c for c in by_recent if c.guid and c.guid == r]
        if exact:
            return exact, None
        by_ident = [c for c in by_recent if c.identifier and c.identifier == r and c.is_group]
        if by_ident:
            return by_ident, None

        contact = self.contacts.get(r)
        contact_name: str | None = None
        if contact is not None:
            norms = self._handles_for_contact(contact)
            contact_name = contact.display_name
        else:
            norm = normalize_handle(r) if ("@" in r or looks_like_phone(r)) else None
            if not norm:
                raise LookupError(
                    f"No conversation found for {r!r}. Pass a chat_id from messages_threads, "
                    "a phone number or email, or a contact id from contacts_search."
                )
            norms = {norm}

        direct = [
            c
            for c in by_recent
            if not c.is_group and any(normalize_handle(h) in norms for h in c.handles)
        ]
        if not direct:
            raise LookupError("No 1:1 conversation found for that contact or handle.")
        return direct, contact_name

    # ---- read --------------------------------------------------------------

    def read(self, chat: str, *, limit: int = 50, before: str | None = None) -> dict[str, Any]:
        """One conversation in chronological order, paginated backwards via `before`."""
        clamped = max(1, min(int(limit), MAX_READ_LIMIT))
        cursor_date: int | None = None
        cursor_rowid: int | None = None
        if before and before.strip():
            b = before.strip()
            if ":" in b and b.split(":", 1)[0].lstrip("-").isdigit():
                d, rid = b.split(":", 1)
                cursor_date, cursor_rowid = int(d), int(rid)
            else:
                cursor_date, cursor_rowid = iso_to_apple_ns(b), -1

        conn = self._open()
        try:
            chats, _contact_name = self._resolve_chats(conn, chat)
            select, extra = self._message_select(conn)
            placeholders = ",".join("?" for _ in chats)
            params: list[Any] = [c.rowid for c in chats]
            where_cursor = ""
            if cursor_date is not None:
                if cursor_rowid is not None and cursor_rowid >= 0:
                    where_cursor = " AND (m.date < ? OR (m.date = ? AND m.ROWID < ?))"
                    params.extend([cursor_date, cursor_date, cursor_rowid])
                else:
                    where_cursor = " AND m.date < ?"
                    params.append(cursor_date)
            params.append(clamped + 1)
            rows = conn.execute(
                f"""
                SELECT {select}
                FROM message m
                JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                LEFT JOIN handle h ON h.ROWID = m.handle_id
                WHERE cmj.chat_id IN ({placeholders}){extra}{where_cursor}
                GROUP BY m.ROWID
                ORDER BY m.date DESC, m.ROWID DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        has_more = len(rows) > clamped
        rows = rows[:clamped]
        messages: list[dict[str, Any]] = []
        for row in reversed(rows):
            text, truncated = _truncate(self._message_text(row), MAX_TEXT_CHARS)
            from_me = bool(row["is_from_me"])
            handle = None if from_me else (row["handle"] or None)
            msg: dict[str, Any] = {
                "id": int(row["rowid"]),
                "time": apple_timestamp_to_iso(row["date_raw"]),
                "from_me": from_me,
                "sender": "me" if from_me else (self._name(handle) or handle or "unknown"),
                "handle": handle,
                "text": text,
            }
            if truncated:
                msg["truncated"] = True
            messages.append(msg)

        primary = chats[0]
        out: dict[str, Any] = {
            "chat": self._chat_summary(primary),
            "count": len(messages),
            "messages": messages,
            "has_more": has_more,
            "next_before": (
                f"{int(rows[-1]['date_raw'] or 0)}:{int(rows[-1]['rowid'])}" if has_more and rows else None
            ),
        }
        if len(chats) > 1:
            out["merged_chat_ids"] = [c.chat_id for c in chats]
        if self.contacts.last_hint:
            out["contacts_hint"] = self.contacts.last_hint
        return out

    # ---- send support ------------------------------------------------------

    def _last_direct_date_by_handle(self, conn: sqlite3.Connection) -> tuple[dict[str, int], dict[str, str]]:
        """Map normalized handle -> latest 1:1 message date, and -> chat.db handle spelling."""
        latest: dict[str, int] = {}
        spelling: dict[str, str] = {}
        for chat in self._load_chats(conn):
            if chat.is_group:
                continue
            for h in chat.handles:
                norm = normalize_handle(h)
                if not norm:
                    continue
                spelling.setdefault(norm, h)
                if _date_ns(chat.last_date) > latest.get(norm, 0):
                    latest[norm] = _date_ns(chat.last_date)
        return latest, spelling

    def resolve_send_target(self, to: str) -> SendTarget:
        """Resolve `to` (phone, email, chat id, or contact id) to one concrete recipient.

        Never guesses: a contact with several handles and no single most-recent 1:1
        conversation raises AmbiguousRecipientError listing the options.
        """
        r = to.strip()
        if not r:
            raise ValueError("`to` must be non-empty")

        conn: sqlite3.Connection | None = None
        try:
            try:
                conn = self._open()
            except (FileNotFoundError, PermissionError):
                conn = None

            if conn is not None:
                chats = self._load_chats(conn)
                for c in chats:
                    if c.guid and c.guid == r:
                        return SendTarget("chat", c.guid, chat_id=c.guid)
                groups = [c for c in chats if c.is_group and c.identifier == r and c.guid]
                if len(groups) == 1:
                    return SendTarget("chat", groups[0].guid, chat_id=groups[0].guid)

            contact = self.contacts.get(r)
            if contact is not None:
                return self._resolve_contact_target(conn, contact)

            if conn is None and ";" in r:
                raise LookupError("Unknown chat id (chat.db unreadable or chat not found).")
            if ";" in r and r.split(";", 1)[0] in {"iMessage", "SMS", "RCS", "any"}:
                raise LookupError(f"Unknown chat id {r!r}; use messages_threads to list chats.")

            email = normalize_email(r) if "@" in r and " " not in r else None
            if email:
                handle = r
                if conn is not None:
                    handle = self._last_direct_date_by_handle(conn)[1].get(email, r)
                return SendTarget("handle", handle, contact_name=self.contacts.lookup_name(handle))
            if looks_like_phone(r) and normalize_phone(r):
                handle = self._clean_phone(r)
                if conn is not None:
                    handle = self._last_direct_date_by_handle(conn)[1].get(normalize_phone(r) or "", handle)
                return SendTarget("handle", handle, contact_name=self.contacts.lookup_name(handle))
            raise LookupError(
                f"Can't resolve recipient {r!r}. Pass a phone number, an email, a chat_id from "
                "messages_threads, or a contact id from contacts_search."
            )
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _clean_phone(raw: str) -> str:
        stripped = raw.strip()
        digits = "".join(ch for ch in stripped if ch.isdigit())
        return ("+" + digits) if stripped.startswith("+") else digits

    def _resolve_contact_target(self, conn: sqlite3.Connection | None, contact: Contact) -> SendTarget:
        handles = contact.handles
        if not handles:
            raise LookupError(f"Contact {contact.id!r} has no phone numbers or emails.")
        latest: dict[str, int] = {}
        spelling: dict[str, str] = {}
        if conn is not None:
            latest, spelling = self._last_direct_date_by_handle(conn)

        def as_target(h: Any) -> SendTarget:
            value = spelling.get(h.normalized) or (
                self._clean_phone(h.value) if h.kind == "phone" else h.value
            )
            return SendTarget("handle", value, contact_name=contact.display_name)

        if len(handles) == 1:
            return as_target(handles[0])

        dated = sorted(
            ((latest.get(h.normalized, 0), h) for h in handles),
            key=lambda dh: dh[0],
            reverse=True,
        )
        if dated[0][0] > 0 and dated[0][0] != dated[1][0]:
            return as_target(dated[0][1])

        options = [
            {
                "kind": h.kind,
                "value": h.value,
                "label": h.label,
                "last_conversation_at": apple_timestamp_to_iso(d) if d else None,
            }
            for d, h in dated
        ]
        raise AmbiguousRecipientError(
            f"{contact.display_name} has {len(handles)} handles and no single most-recent "
            "conversation; call messages_send again with `to` set to one of these values.",
            options,
        )

    def max_message_rowid(self) -> int:
        conn = self._open()
        try:
            row = conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()
            return int(row[0])
        finally:
            conn.close()

    def find_outgoing(self, after_rowid: int, text: str, target: SendTarget) -> dict[str, Any] | None:
        """Find our just-sent outgoing row (ROWID > after_rowid, from me, same text and target)."""
        conn = self._open()
        try:
            select, _extra = self._message_select(conn)
            chat_cols = self._cols(conn, "chat")
            guid_sel = "c.guid" if "guid" in chat_cols else "c.chat_identifier"
            rows = conn.execute(
                f"""
                SELECT {select}, cmj.chat_id AS chat_rowid, {guid_sel} AS chat_guid
                FROM message m
                LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                LEFT JOIN chat c ON c.ROWID = cmj.chat_id
                LEFT JOIN handle h ON h.ROWID = m.handle_id
                WHERE m.ROWID > ? AND m.is_from_me = 1
                ORDER BY m.ROWID ASC
                """,
                (after_rowid,),
            ).fetchall()
            chats = {c.rowid: c for c in self._load_chats(conn)}
        finally:
            conn.close()

        want_text = text.strip()
        want_norm = normalize_handle(target.value) if target.kind == "handle" else None
        for row in rows:
            if self._message_text(row).strip() != want_text:
                continue
            chat = chats.get(int(row["chat_rowid"])) if row["chat_rowid"] is not None else None
            if target.kind == "chat":
                if not chat or chat.chat_id != target.value:
                    continue
            else:
                handles = list(chat.handles) if chat else []
                if row["handle"]:
                    handles.append(str(row["handle"]))
                if chat is not None and chat.is_group:
                    continue
                if not any(normalize_handle(h) == want_norm for h in handles):
                    continue
            return {
                "message_id": int(row["rowid"]),
                "chat_id": chat.chat_id if chat else None,
                "service": row["service"],
                "error": int(row["error"] or 0),
            }
        return None
