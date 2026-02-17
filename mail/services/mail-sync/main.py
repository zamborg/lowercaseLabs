import argparse
import email
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email import policy
from email.header import decode_header
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import imaplib
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from packages.shared.crypto import encrypt_value
from db import Database


def _load_fixtures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"fixtures not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync emails into the database")
    parser.add_argument(
        "--source",
        choices=["auto", "fixtures", "imap"],
        default="auto",
        help="Data source to use",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default=str(ROOT / "services" / "mail-sync" / "fixtures" / "sample_emails.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--since-days", type=int, default=None)
    parser.add_argument("--mailbox", type=str, default=None)
    parser.add_argument("--unseen-only", action="store_true")
    return parser.parse_args()


def _decode_header(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parts = decode_header(value)
    decoded: list[str] = []
    for text, charset in parts:
        if isinstance(text, bytes):
            decoded.append(text.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(text)
    result = "".join(decoded).strip()
    return result or None


def _parse_addresses(value: Optional[str]) -> list[dict[str, str]]:
    if not value:
        return []
    addresses = []
    for name, addr in getaddresses([value]):
        if not addr:
            continue
        clean_name = _decode_header(name)
        addresses.append({"name": clean_name or "", "email": addr})
    return addresses


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if not parsed:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
    if not text:
        return text
    return text[:limit]


def _extract_bodies(message: email.message.Message, limit: int) -> tuple[Optional[str], Optional[str]]:
    body_text: Optional[str] = None
    body_html: Optional[str] = None

    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                continue
            content_disposition = part.get("Content-Disposition", "")
            if content_disposition and "attachment" in content_disposition.lower():
                continue
            content_type = part.get_content_type()
            try:
                payload = part.get_content()
            except (LookupError, ValueError):
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
            if content_type == "text/plain" and body_text is None:
                body_text = str(payload)
            elif content_type == "text/html" and body_html is None:
                body_html = str(payload)
    else:
        try:
            payload = message.get_content()
        except (LookupError, ValueError):
            payload = message.get_payload(decode=True)
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
        if message.get_content_type() == "text/html":
            body_html = str(payload)
        else:
            body_text = str(payload)

    body_text = _truncate(body_text, limit)
    body_html = _truncate(body_html, limit)
    return body_text, body_html


def _build_snippet(body_text: Optional[str], body_html: Optional[str], subject: Optional[str]) -> str:
    base = body_text
    if not base and body_html:
        base = _strip_html(body_html)
    if not base:
        base = subject or ""
    snippet = re.sub(r"\s+", " ", base).strip()
    return snippet[:160]


def _build_thread_key(message: email.message.Message, message_id: str) -> str:
    in_reply_to = message.get("In-Reply-To")
    if in_reply_to:
        return in_reply_to.strip()
    references = message.get("References")
    if references:
        refs = references.split()
        if refs:
            return refs[-1]
    return message_id


def _parse_flags(meta: bytes) -> list[str]:
    if not meta:
        return []
    match = re.search(rb"FLAGS \(([^)]*)\)", meta)
    if not match:
        return []
    raw = match.group(1).decode("utf-8", errors="ignore")
    return [flag for flag in raw.split() if flag]


def _fetch_message(imap: imaplib.IMAP4, msg_id: bytes) -> tuple[Optional[bytes], list[str]]:
    typ, data = imap.fetch(msg_id, "(RFC822 FLAGS)")
    if typ != "OK":
        return None, []
    raw_message = None
    flags: list[str] = []
    for item in data:
        if isinstance(item, tuple):
            raw_message = item[1]
            flags = _parse_flags(item[0])
    return raw_message, flags


def _imap_search(
    imap: imaplib.IMAP4,
    since_days: Optional[int],
    unseen_only: bool,
) -> list[bytes]:
    criteria: list[str] = ["UNSEEN" if unseen_only else "ALL"]
    if since_days and since_days > 0:
        since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%d-%b-%Y")
        criteria.extend(["SINCE", since_date])
    typ, data = imap.search(None, *criteria)
    if typ != "OK" or not data:
        return []
    return data[0].split()


def _parse_message(raw_message: bytes, flags: list[str], body_limit: int) -> dict[str, Any]:
    message = email.message_from_bytes(raw_message, policy=policy.default)

    message_id = _decode_header(message.get("Message-ID"))
    if not message_id:
        digest = hashlib.sha256(raw_message).hexdigest()[:16]
        message_id = f"<local-{digest}>"

    subject = _decode_header(message.get("Subject"))
    body_text, body_html = _extract_bodies(message, body_limit)
    date_value = _parse_date(message.get("Date"))
    thread_key = _build_thread_key(message, message_id)

    return {
        "message_id": message_id,
        "thread_key": thread_key,
        "from": _parse_addresses(message.get("From")),
        "to": _parse_addresses(message.get("To")),
        "cc": _parse_addresses(message.get("Cc")),
        "bcc": _parse_addresses(message.get("Bcc")),
        "subject": subject,
        "date": date_value,
        "snippet": _build_snippet(body_text, body_html, subject),
        "body_text": body_text,
        "body_html": body_html,
        "labels": {"imap_flags": flags},
        "is_read": "\\Seen" in flags,
    }


def _sync_imap(
    db: Database,
    account_id: str,
    host: str,
    port: int,
    user: str,
    password: str,
    mailbox: str,
    limit: int,
    since_days: Optional[int],
    unseen_only: bool,
    use_ssl: bool,
    body_limit: int,
) -> int:
    if use_ssl:
        imap: imaplib.IMAP4 = imaplib.IMAP4_SSL(host, port)
    else:
        imap = imaplib.IMAP4(host, port)

    imap.login(user, password)
    imap.select(mailbox)

    try:
        message_ids = _imap_search(imap, since_days, unseen_only)
        if limit > 0:
            message_ids = message_ids[-limit:]
        inserted = 0
        for msg_id in message_ids:
            raw_message, flags = _fetch_message(imap, msg_id)
            if not raw_message:
                continue
            payload = _parse_message(raw_message, flags, body_limit)
            if db.email_exists(account_id, payload.get("message_id")):
                continue
            db.insert_email(account_id, payload)
            inserted += 1
        return inserted
    finally:
        imap.logout()


def _sync_fixtures(db: Database, account_id: str, fixtures_path: Path) -> int:
    fixtures = _load_fixtures(fixtures_path)
    inserted = 0
    for item in fixtures:
        if db.email_exists(account_id, item.get("message_id")):
            continue
        db.insert_email(account_id, item)
        inserted += 1
    return inserted


def _resolve_source(args: argparse.Namespace, imap_configured: bool) -> str:
    if args.source != "auto":
        return args.source
    return "imap" if imap_configured else "fixtures"


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    args = _parse_args()
    load_dotenv()

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/agentic_mail",
    )
    dev_user_email = os.environ.get("DEV_USER_EMAIL", "dev@local")
    dev_account_email = os.environ.get("DEV_ACCOUNT_EMAIL", dev_user_email)
    encryption_key = os.environ.get("ENCRYPTION_KEY", "dev-insecure-key")

    imap_host = os.environ.get("IMAP_HOST")
    imap_user = os.environ.get("IMAP_USER")
    imap_pass = os.environ.get("IMAP_PASSWORD")
    imap_port = int(os.environ.get("IMAP_PORT", "993"))
    imap_mailbox = os.environ.get("IMAP_MAILBOX", "INBOX")
    imap_limit = int(os.environ.get("IMAP_LIMIT", "200"))
    imap_since_days = int(os.environ.get("IMAP_SINCE_DAYS", "30"))
    imap_unseen_only = _get_env_bool("IMAP_UNSEEN_ONLY", False)
    imap_use_ssl = _get_env_bool("IMAP_USE_SSL", True)
    body_limit = int(os.environ.get("IMAP_BODY_LIMIT", "20000"))

    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")

    if args.limit is not None:
        imap_limit = args.limit
    if args.since_days is not None:
        imap_since_days = args.since_days
    if args.mailbox is not None:
        imap_mailbox = args.mailbox
    if args.unseen_only:
        imap_unseen_only = True

    imap_configured = bool(imap_host and imap_user and imap_pass)
    source = _resolve_source(args, imap_configured)

    db = Database(database_url)
    db.init_schema()

    user = db.ensure_user(dev_user_email)
    account = db.ensure_account(
        user["id"],
        dev_account_email,
        imap_host=imap_host,
        imap_user=imap_user,
        imap_encrypted_pass=encrypt_value(imap_pass, encryption_key) if imap_pass else None,
        smtp_host=smtp_host,
        smtp_user=smtp_user,
        smtp_encrypted_pass=encrypt_value(smtp_pass, encryption_key) if smtp_pass else None,
    )

    if source == "fixtures":
        inserted = _sync_fixtures(db, account["id"], Path(args.fixtures))
        print(f"Inserted {inserted} fixture emails")
        return

    if not imap_configured:
        raise RuntimeError("IMAP is not configured. Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD.")

    inserted = _sync_imap(
        db,
        account["id"],
        imap_host,
        imap_port,
        imap_user,
        imap_pass,
        imap_mailbox,
        imap_limit,
        imap_since_days,
        imap_unseen_only,
        imap_use_ssl,
        body_limit,
    )
    print(f"Inserted {inserted} IMAP emails")


if __name__ == "__main__":
    main()
