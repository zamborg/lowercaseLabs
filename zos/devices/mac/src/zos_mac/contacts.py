"""Read-only macOS AddressBook contact lookup with mtime-invalidated cache.

Never logs contact names, phone numbers, or email addresses.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

DEFAULT_ADDRESSBOOK_ROOT = Path("~/Library/Application Support/AddressBook")
_NON_DIGITS_RE = re.compile(r"\D+")


def normalize_phone(raw: str | None) -> str | None:
    """Normalize a phone handle to its last 10 digits (or 7-9 digits if shorter)."""
    if not raw:
        return None
    digits = _NON_DIGITS_RE.sub("", raw)
    if len(digits) >= 10:
        return digits[-10:]
    if len(digits) >= 7:
        return digits
    return None


def normalize_email(raw: str | None) -> str | None:
    if not raw or "@" not in raw:
        return None
    cleaned = raw.strip().lower()
    return cleaned or None


def normalize_handle(raw: str | None) -> str | None:
    if not raw:
        return None
    if "@" in raw:
        return normalize_email(raw)
    return normalize_phone(raw)


def _display_name(
    first: str | None,
    last: str | None,
    org: str | None,
    nick: str | None,
) -> str | None:
    parts = [p.strip() for p in (first, last) if p and p.strip()]
    if parts:
        return " ".join(parts)
    if org and org.strip():
        return org.strip()
    if nick and nick.strip():
        return nick.strip()
    return None


class ContactsIndex:
    """Read-only cache over macOS AddressBook-v22.abcddb files with mtime invalidation."""

    def __init__(self, addressbook_root: Path | None = None) -> None:
        self.root = (addressbook_root or DEFAULT_ADDRESSBOOK_ROOT).expanduser()
        self._cached_snapshot: tuple[tuple[str, int, int], ...] | None = None
        self._handle_to_name: dict[str, str] = {}
        self._contacts: list[tuple[str, set[str], set[str]]] = []
        self._last_hint: str | None = None

    @property
    def last_hint(self) -> str | None:
        return self._last_hint

    def _discover_dbs(self) -> tuple[list[Path], bool]:
        """Return (sorted list of abcddb paths, had_permission_error)."""
        paths: list[Path] = []
        perm_err = False
        try:
            if not self.root.exists():
                return [], False
            sources_dir = self.root / "Sources"
            if sources_dir.is_dir():
                for p in sorted(sources_dir.glob("*/AddressBook-v22.abcddb")):
                    paths.append(p)
            top = self.root / "AddressBook-v22.abcddb"
            if top.is_file():
                paths.append(top)
        except PermissionError:
            perm_err = True
        return paths, perm_err

    def _snapshot(self, db_paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
        snap: list[tuple[str, int, int]] = []
        for p in db_paths:
            try:
                st = p.stat()
                snap.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                snap.append((str(p), -1, -1))
        return tuple(snap)

    def _ensure_loaded(self) -> None:
        db_paths, perm_err = self._discover_dbs()
        if perm_err:
            self._handle_to_name = {}
            self._contacts = []
            self._cached_snapshot = None
            resolved_py = os.path.realpath(sys.executable)
            self._last_hint = (
                f"Contacts unavailable (AddressBook permission denied). "
                f"Grant Full Disk Access to {resolved_py} to resolve contact names."
            )
            return

        if not db_paths:
            self._handle_to_name = {}
            self._contacts = []
            self._cached_snapshot = ()
            self._last_hint = None
            return

        snap = self._snapshot(db_paths)
        if self._cached_snapshot == snap:
            return

        handle_to_name: dict[str, str] = {}
        # Aggregate handles by display name (case-folded key -> (display_name, phones, emails))
        by_name: dict[str, tuple[str, set[str], set[str]]] = {}
        any_error = False
        any_loaded = False

        for db_path in db_paths:
            try:
                self._load_single_db(db_path, handle_to_name, by_name)
                any_loaded = True
            except (PermissionError, sqlite3.OperationalError, sqlite3.DatabaseError, OSError):
                any_error = True

        self._handle_to_name = handle_to_name
        self._contacts = [(disp, phones, emails) for disp, phones, emails in by_name.values()]
        self._cached_snapshot = snap if not any_error else None

        if any_error and not any_loaded:
            resolved_py = os.path.realpath(sys.executable)
            self._last_hint = (
                f"Contacts unavailable (AddressBook database unreadable). "
                f"Grant Full Disk Access to {resolved_py} to resolve contact names."
            )
        elif any_error:
            self._last_hint = "Some AddressBook sources were unreadable; contact names may be incomplete."
        else:
            self._last_hint = None

    def _load_single_db(
        self,
        db_path: Path,
        handle_to_name: dict[str, str],
        by_name: dict[str, tuple[str, set[str], set[str]]],
    ) -> None:
        uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 1000")
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "ZABCDRECORD" not in tables:
                return

            rec_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(ZABCDRECORD)").fetchall()
            }
            sel_first = "ZFIRSTNAME" if "ZFIRSTNAME" in rec_cols else "NULL AS ZFIRSTNAME"
            sel_last = "ZLASTNAME" if "ZLASTNAME" in rec_cols else "NULL AS ZLASTNAME"
            sel_org = "ZORGANIZATION" if "ZORGANIZATION" in rec_cols else "NULL AS ZORGANIZATION"
            sel_nick = "ZNICKNAME" if "ZNICKNAME" in rec_cols else "NULL AS ZNICKNAME"

            pk_to_name: dict[int, str] = {}
            for row in conn.execute(
                f"SELECT Z_PK, {sel_first}, {sel_last}, {sel_org}, {sel_nick} FROM ZABCDRECORD"
            ):
                disp = _display_name(
                    row["ZFIRSTNAME"],
                    row["ZLASTNAME"],
                    row["ZORGANIZATION"],
                    row["ZNICKNAME"],
                )
                if disp:
                    pk_to_name[int(row["Z_PK"])] = disp

            def _ensure_entry(disp: str) -> tuple[set[str], set[str]]:
                key = disp.lower()
                if key not in by_name:
                    by_name[key] = (disp, set(), set())
                return by_name[key][1], by_name[key][2]

            if "ZABCDPHONENUMBER" in tables:
                for row in conn.execute("SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER"):
                    owner = row["ZOWNER"]
                    if owner is None or int(owner) not in pk_to_name:
                        continue
                    norm = normalize_phone(row["ZFULLNUMBER"])
                    if not norm:
                        continue
                    disp = pk_to_name[int(owner)]
                    handle_to_name.setdefault(norm, disp)
                    phones, _emails = _ensure_entry(disp)
                    phones.add(norm)

            if "ZABCDEMAILADDRESS" in tables:
                email_cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(ZABCDEMAILADDRESS)").fetchall()
                }
                email_expr = (
                    "COALESCE(ZADDRESSNORMALIZED, ZADDRESS)"
                    if "ZADDRESSNORMALIZED" in email_cols and "ZADDRESS" in email_cols
                    else ("ZADDRESS" if "ZADDRESS" in email_cols else "ZADDRESSNORMALIZED")
                )
                for row in conn.execute(
                    f"SELECT ZOWNER, {email_expr} AS addr FROM ZABCDEMAILADDRESS"
                ):
                    owner = row["ZOWNER"]
                    if owner is None or int(owner) not in pk_to_name:
                        continue
                    norm = normalize_email(row["addr"])
                    if not norm:
                        continue
                    disp = pk_to_name[int(owner)]
                    handle_to_name.setdefault(norm, disp)
                    _phones, emails = _ensure_entry(disp)
                    emails.add(norm)
        finally:
            conn.close()

    def lookup_name(self, handle: str | None) -> str | None:
        """Resolve a phone number or email handle to a contact display name, or None."""
        self._ensure_loaded()
        norm = normalize_handle(handle)
        if not norm:
            return None
        return self._handle_to_name.get(norm)

    def resolve_handles_for_name(self, query: str) -> tuple[set[str], set[str]]:
        """Resolve a partial or full contact name to (phone_last10s, lowercase_emails)."""
        self._ensure_loaded()
        q = query.strip().lower()
        if not q:
            return set(), set()
        phones: set[str] = set()
        emails: set[str] = set()
        for disp, c_phones, c_emails in self._contacts:
            if q in disp.lower():
                phones.update(c_phones)
                emails.update(c_emails)
        return phones, emails
