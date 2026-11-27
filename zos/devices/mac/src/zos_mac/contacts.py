"""Read-only macOS AddressBook contact lookup with mtime-invalidated cache.

Reads the AddressBook SQLite stores directly (covered by Full Disk Access), never the
Contacts framework. Never logs contact names, phone numbers, or email addresses.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_ADDRESSBOOK_ROOT = Path("~/Library/Application Support/AddressBook")
_NON_DIGITS_RE = re.compile(r"\D+")
_LABEL_RE = re.compile(r"^_\$!<(.+)>!\$_$")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
FUZZY_MIN_RATIO = 0.75
MAX_CONTACT_RESULTS = 50


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


def clean_label(raw: str | None) -> str | None:
    """Turn AddressBook labels like `_$!<Mobile>!$_` into `mobile`; keep custom labels verbatim."""
    if not raw or not raw.strip():
        return None
    m = _LABEL_RE.match(raw.strip())
    if m:
        return m.group(1).strip().lower()
    return raw.strip()


@dataclass(slots=True)
class ContactHandle:
    kind: str  # "phone" | "email"
    value: str  # raw value as stored in AddressBook
    normalized: str
    label: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label}


@dataclass(slots=True)
class Contact:
    id: str
    display_name: str
    first_name: str | None = None
    last_name: str | None = None
    nickname: str | None = None
    organization: str | None = None
    phones: list[ContactHandle] = field(default_factory=list)
    emails: list[ContactHandle] = field(default_factory=list)

    @property
    def handles(self) -> list[ContactHandle]:
        return [*self.phones, *self.emails]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "display_name": self.display_name}
        if self.nickname:
            out["nickname"] = self.nickname
        if self.organization:
            out["organization"] = self.organization
        out["phones"] = [h.to_dict() for h in self.phones]
        out["emails"] = [h.to_dict() for h in self.emails]
        return out


def _words(*texts: str | None) -> list[str]:
    out: list[str] = []
    for text in texts:
        if text:
            out.extend(w.lower() for w in _WORD_RE.findall(text))
    return out


def _token_score(token: str, words: list[str]) -> float:
    """Score one query token against a contact's words (exact > prefix > substring > typo)."""
    best = 0.0
    for w in words:
        if w == token:
            return 100.0
        if w.startswith(token):
            best = max(best, 80.0)
        elif len(token) >= 3 and token in w:
            best = max(best, 55.0)
        elif len(token) >= 3:
            ratio = SequenceMatcher(None, token, w).ratio()
            if ratio >= FUZZY_MIN_RATIO:
                best = max(best, 50.0 * ratio)
    return best


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


_PHONE_CHARS_RE = re.compile(r"^\+?[\d\s().\-]{7,}$")


def looks_like_phone(value: str) -> bool:
    return bool(_PHONE_CHARS_RE.match(value.strip()))


def _looks_like_handle(value: str) -> bool:
    v = value.strip()
    return ("@" in v and " " not in v) or looks_like_phone(v)


def _source_tag(db_path: Path) -> str:
    """Stable short tag for an AddressBook source db (its Sources/<UUID> dir, or `local`)."""
    if db_path.parent.parent.name == "Sources":
        return db_path.parent.name
    return "local"


class ContactsIndex:
    """Read-only cache over macOS AddressBook-v22.abcddb files with mtime invalidation."""

    def __init__(self, addressbook_root: Path | None = None) -> None:
        self.root = (addressbook_root or DEFAULT_ADDRESSBOOK_ROOT).expanduser()
        self._cached_snapshot: tuple[tuple[str, int, int], ...] | None = None
        self._handle_to_name: dict[str, str] = {}
        self._contacts: list[tuple[str, set[str], set[str]]] = []
        self._records: dict[str, Contact] = {}
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

    def _reset(self, snapshot: tuple[tuple[str, int, int], ...] | None) -> None:
        self._handle_to_name = {}
        self._contacts = []
        self._records = {}
        self._cached_snapshot = snapshot

    def _ensure_loaded(self) -> None:
        db_paths, perm_err = self._discover_dbs()
        if perm_err:
            self._reset(None)
            resolved_py = os.path.realpath(sys.executable)
            self._last_hint = (
                f"Contacts unavailable (AddressBook permission denied). "
                f"Grant Full Disk Access to {resolved_py} to resolve contact names."
            )
            return

        if not db_paths:
            self._reset(())
            self._last_hint = None
            return

        snap = self._snapshot(db_paths)
        if self._cached_snapshot == snap:
            return

        handle_to_name: dict[str, str] = {}
        # Aggregate handles by display name (case-folded key -> (display_name, phones, emails))
        by_name: dict[str, tuple[str, set[str], set[str]]] = {}
        records: dict[str, Contact] = {}
        any_error = False
        any_loaded = False

        for db_path in db_paths:
            try:
                self._load_single_db(db_path, handle_to_name, by_name, records)
                any_loaded = True
            except (PermissionError, sqlite3.OperationalError, sqlite3.DatabaseError, OSError):
                any_error = True

        self._handle_to_name = handle_to_name
        self._contacts = [(disp, phones, emails) for disp, phones, emails in by_name.values()]
        self._records = records
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
        records: dict[str, Contact],
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

            def _cols(table: str) -> set[str]:
                return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

            def _sel(cols: set[str], name: str) -> str:
                return name if name in cols else f"NULL AS {name}"

            rec_cols = _cols("ZABCDRECORD")
            sel = ", ".join(
                _sel(rec_cols, c)
                for c in ("ZFIRSTNAME", "ZLASTNAME", "ZORGANIZATION", "ZNICKNAME", "ZUNIQUEID")
            )
            tag = _source_tag(db_path)

            pk_to_contact: dict[int, Contact] = {}
            for row in conn.execute(f"SELECT Z_PK, {sel} FROM ZABCDRECORD"):
                disp = _display_name(
                    row["ZFIRSTNAME"],
                    row["ZLASTNAME"],
                    row["ZORGANIZATION"],
                    row["ZNICKNAME"],
                )
                if not disp:
                    continue
                pk = int(row["Z_PK"])
                unique = row["ZUNIQUEID"]
                cid = str(unique) if unique else f"{tag}:{pk}"
                contact = Contact(
                    id=cid,
                    display_name=disp,
                    first_name=row["ZFIRSTNAME"],
                    last_name=row["ZLASTNAME"],
                    nickname=row["ZNICKNAME"],
                    organization=row["ZORGANIZATION"],
                )
                pk_to_contact[pk] = contact
                records.setdefault(cid, contact)

            def _ensure_entry(disp: str) -> tuple[set[str], set[str]]:
                key = disp.lower()
                if key not in by_name:
                    by_name[key] = (disp, set(), set())
                return by_name[key][1], by_name[key][2]

            if "ZABCDPHONENUMBER" in tables:
                ph_cols = _cols("ZABCDPHONENUMBER")
                order = "ZORDERINGINDEX, Z_PK" if "ZORDERINGINDEX" in ph_cols else "Z_PK"
                for row in conn.execute(
                    f"SELECT ZOWNER, ZFULLNUMBER, {_sel(ph_cols, 'ZLABEL')} "
                    f"FROM ZABCDPHONENUMBER ORDER BY {order}"
                ):
                    owner = row["ZOWNER"]
                    if owner is None or int(owner) not in pk_to_contact:
                        continue
                    norm = normalize_phone(row["ZFULLNUMBER"])
                    if not norm:
                        continue
                    contact = pk_to_contact[int(owner)]
                    handle_to_name.setdefault(norm, contact.display_name)
                    phones, _emails = _ensure_entry(contact.display_name)
                    phones.add(norm)
                    if all(h.normalized != norm for h in contact.phones):
                        contact.phones.append(
                            ContactHandle(
                                "phone", str(row["ZFULLNUMBER"]).strip(), norm, clean_label(row["ZLABEL"])
                            )
                        )

            if "ZABCDEMAILADDRESS" in tables:
                email_cols = _cols("ZABCDEMAILADDRESS")
                email_expr = (
                    "COALESCE(ZADDRESSNORMALIZED, ZADDRESS)"
                    if "ZADDRESSNORMALIZED" in email_cols and "ZADDRESS" in email_cols
                    else ("ZADDRESS" if "ZADDRESS" in email_cols else "ZADDRESSNORMALIZED")
                )
                raw_expr = "ZADDRESS" if "ZADDRESS" in email_cols else email_expr
                order = "ZORDERINGINDEX, Z_PK" if "ZORDERINGINDEX" in email_cols else "Z_PK"
                for row in conn.execute(
                    f"SELECT ZOWNER, {email_expr} AS addr, {raw_expr} AS raw_addr, "
                    f"{_sel(email_cols, 'ZLABEL')} FROM ZABCDEMAILADDRESS ORDER BY {order}"
                ):
                    owner = row["ZOWNER"]
                    if owner is None or int(owner) not in pk_to_contact:
                        continue
                    norm = normalize_email(row["addr"])
                    if not norm:
                        continue
                    contact = pk_to_contact[int(owner)]
                    handle_to_name.setdefault(norm, contact.display_name)
                    _phones, emails = _ensure_entry(contact.display_name)
                    emails.add(norm)
                    if all(h.normalized != norm for h in contact.emails):
                        contact.emails.append(
                            ContactHandle(
                                "email",
                                str(row["raw_addr"] or norm).strip(),
                                norm,
                                clean_label(row["ZLABEL"]),
                            )
                        )
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

    def get(self, contact_id: str) -> Contact | None:
        """Return one contact by id (as returned by `search`), or None."""
        self._ensure_loaded()
        return self._records.get(contact_id.strip()) if contact_id else None

    def find_by_handle(self, handle: str) -> list[Contact]:
        """All contacts owning a phone/email handle (normalized comparison)."""
        self._ensure_loaded()
        norm = normalize_handle(handle)
        if not norm:
            return []
        return [c for c in self._records.values() if any(h.normalized == norm for h in c.handles)]

    def search(self, query: str, *, limit: int = 10) -> list[tuple[Contact, float]]:
        """Fuzzy match over name, nickname, and organization (or exact phone/email handle).

        Every query token must match some word (exact, prefix, substring, or close typo).
        Results are sorted by score (best first), then display name.
        """
        self._ensure_loaded()
        q = query.strip()
        if not q:
            raise ValueError("query must be non-empty")
        clamped = max(1, min(int(limit), MAX_CONTACT_RESULTS))

        scored: list[tuple[Contact, float]] = []
        norm_handle = normalize_handle(q) if _looks_like_handle(q) else None
        tokens = _words(q)
        for contact in self._records.values():
            if norm_handle and any(h.normalized == norm_handle for h in contact.handles):
                scored.append((contact, 1000.0))
                continue
            if not tokens:
                continue
            words = _words(
                contact.display_name,
                contact.first_name,
                contact.last_name,
                contact.nickname,
                contact.organization,
            )
            total = 0.0
            for tok in tokens:
                s = _token_score(tok, words)
                if s == 0.0:
                    total = 0.0
                    break
                total += s
            if total > 0.0:
                scored.append((contact, total / len(tokens)))

        scored.sort(key=lambda cs: (-cs[1], cs[0].display_name.lower(), cs[0].id))
        return scored[:clamped]
