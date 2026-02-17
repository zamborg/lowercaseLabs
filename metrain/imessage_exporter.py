#!/usr/bin/env python3
"""
Export iMessage conversations to OpenAI fine-tuning JSONL.

By default this only writes messages that were sent by the user (is_from_me).
Use --include-context to preface each user message with the most recent response
from another participant so you can fine-tune on a conversation-style prompt.
The exported format now matches the chat-style expectations (every example has a
"messages" array).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence


DEFAULT_DB = Path.home() / "Library" / "Messages" / "chat.db"
DEFAULT_OUTPUT = Path("imessage_export.jsonl")
MAC_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract your iMessage text and save it as an OpenAI fine-tune JSONL file."
    )
    parser.add_argument(
        "db_path",
        nargs="?",
        help="Path to the iMessage chat.db (defaults to ~/Library/Messages/chat.db).",
        default=str(DEFAULT_DB),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the JSONL export (default: imessage_export.jsonl).",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="If set, the most recent incoming reply(s) become the preceding `user` entries before your message.",
    )
    parser.add_argument(
        "--context-depth",
        type=int,
        default=1,
        help="Number of prior incoming messages to include as `user` entries when --include-context is enabled.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of exported user messages (0 = no limit). Useful for quick sanity checks.",
    )
    return parser.parse_args()


def connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"iMessage database not found at {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def mac_timestamp_to_datetime(value: Optional[int]) -> Optional[datetime]:
    if value is None:
        return None
    seconds = float(value)
    if abs(seconds) > 1e12:
        seconds /= 1_000_000_000
    return MAC_EPOCH + timedelta(seconds=seconds)


def load_handles(conn: sqlite3.Connection) -> Mapping[int, str]:
    cursor = conn.execute("SELECT ROWID, id FROM handle")
    return {row[0]: row[1] or "Unknown" for row in cursor}


def iter_message_rows(conn: sqlite3.Connection) -> Iterator[tuple[int, int, int, int, int, Optional[str]]]:
    query = """
    SELECT
        cm.chat_id,
        m.ROWID,
        m.date,
        m.is_from_me,
        m.handle_id,
        m.text
    FROM chat_message_join cm
    JOIN message m ON m.ROWID = cm.message_id
    ORDER BY cm.chat_id, m.date, m.ROWID
    """
    cursor = conn.execute(query)
    for chat_id, rowid, date, is_from_me, handle_id, text in cursor:
        yield chat_id, rowid, date, is_from_me, handle_id or 0, text


def group_messages(
    rows: Iterable[tuple[int, int, int, int, int, Optional[str]]]
) -> Mapping[int, list[dict]]:
    chats: dict[int, list[dict]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for chat_id, rowid, date, is_from_me, handle_id, text in rows:
        key = (chat_id, rowid)
        if key in seen:
            continue
        seen.add(key)
        chats[chat_id].append(
            {
                "rowid": rowid,
                "date": mac_timestamp_to_datetime(date),
                "is_from_me": bool(is_from_me),
                "handle_id": handle_id,
                "text": text,
            }
        )
    return chats


def build_context_messages(
    messages: Sequence[dict], index: int, handles: Mapping[int, str], depth: int
) -> list[dict[str, str]]:
    if depth < 1:
        return []
    context_parts: list[dict[str, str]] = []
    cursor = index - 1
    while cursor >= 0 and len(context_parts) < depth:
        candidate = messages[cursor]
        if not candidate["is_from_me"] and candidate["text"]:
            sender = handles.get(candidate["handle_id"], "Other")
            cleaned = candidate["text"].strip()
            if cleaned:
                context_parts.append({"role": "user", "content": f"{sender}: {cleaned}"})
        cursor -= 1
    context_parts.reverse()
    return context_parts


def export_entries(
    chats: Mapping[int, list[dict]],
    handles: Mapping[int, str],
    include_context: bool,
    context_depth: int,
    limit: int,
) -> Iterator[dict]:
    written = 0
    for chat_id, messages in chats.items():
        for index, message in enumerate(messages):
            if limit and written >= limit:
                return
            text = message["text"]
            if not text or not text.strip():
                continue
            if not message["is_from_me"]:
                continue
            messages_list: list[dict[str, str]] = []
            if include_context:
                messages_list.extend(
                    build_context_messages(messages, index, handles, context_depth)
                )
            messages_list.append({"role": "assistant", "content": text.strip()})
            entry = {"messages": messages_list}
            written += 1
            yield entry


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    with connect(args.db_path) as conn:
        handles = load_handles(conn)
        chats = group_messages(iter_message_rows(conn))
        entries = export_entries(
            chats,
            handles,
            include_context=args.include_context,
            context_depth=max(1, args.context_depth),
            limit=args.limit,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            for entry in entries:
                outfile.write(json.dumps(entry, ensure_ascii=False))
                outfile.write("\n")
    print(f"Wrote export to {output_path.resolve()}")


if __name__ == "__main__":
    main()
