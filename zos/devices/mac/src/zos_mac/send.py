"""Send iMessage/SMS through Messages.app via `osascript`, then confirm via chat.db.

The message text and recipient are passed to AppleScript strictly as argv items
(`on run argv`), never interpolated into script source, so quotes, backslashes,
newlines, and unicode cannot change the script. Never logs text or recipients.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from zos_mac.chatdb import ChatStore, SendTarget

MAX_SEND_CHARS = 4000
DEFAULT_CONFIRM_TIMEOUT_S = 5.0
DEFAULT_POLL_INTERVAL_S = 0.5
OSASCRIPT_TIMEOUT_S = 30.0

OsascriptRunner = Callable[[Sequence[str]], Awaitable[tuple[int, str, str]]]

# argv: 1 = target kind ("chat" | "handle"), 2 = chat guid or handle, 3 = text, 4 = service
SEND_SCRIPT_LINES: tuple[str, ...] = (
    "on run argv",
    "set targetKind to item 1 of argv",
    "set targetValue to item 2 of argv",
    "set messageText to item 3 of argv",
    "set serviceName to item 4 of argv",
    'tell application "Messages"',
    'if targetKind is "chat" then',
    "send messageText to chat id targetValue",
    "else",
    'if serviceName is "SMS" then',
    "set targetAccount to first account whose service type is SMS",
    "else",
    "set targetAccount to first account whose service type is iMessage",
    "end if",
    "send messageText to participant targetValue of targetAccount",
    "end if",
    "end tell",
    "end run",
)

_TCC_MARKERS = ("-1743", "not authorized", "not allowed to send apple events")


def build_osascript_argv(script_lines: Sequence[str], args: Sequence[str]) -> list[str]:
    """Build an `osascript` argv: one `-e` per fixed script line, then data as plain argv.

    The first data argument must not start with `-` so osascript stops option parsing.
    """
    if args and args[0].startswith("-"):
        raise ValueError("first osascript argument must not start with '-'")
    argv = ["osascript"]
    for line in script_lines:
        argv.extend(["-e", line])
    argv.extend(args)
    return argv


async def run_osascript(argv: Sequence[str]) -> tuple[int, str, str]:
    """Default runner: execute osascript without a shell, with a hard timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=OSASCRIPT_TIMEOUT_S)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return 124, "", "osascript timed out"
    return (
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


def automation_hint() -> str:
    return (
        "macOS blocked sending (Automation permission). Allow it in System Settings -> "
        "Privacy & Security -> Automation: enable 'Messages' for "
        f"{os.path.realpath(sys.executable)}."
    )


def validate_send_args(to: Any, text: Any) -> tuple[str, str]:
    to_s = str(to or "").strip()
    text_s = str(text or "")
    if not to_s:
        raise ValueError("`to` must be non-empty")
    if not text_s.strip():
        raise ValueError("`text` must be non-empty")
    if len(text_s) > MAX_SEND_CHARS:
        raise ValueError(f"`text` is too long ({len(text_s)} chars; max {MAX_SEND_CHARS})")
    if "\x00" in text_s or "\x00" in to_s:
        raise ValueError("`to` and `text` must not contain NUL characters")
    return to_s, text_s


class MessageSender:
    """Resolves a recipient, sends via Messages.app, and confirms the outgoing chat.db row."""

    def __init__(
        self,
        store: ChatStore,
        *,
        runner: OsascriptRunner | None = None,
        confirm_timeout_s: float = DEFAULT_CONFIRM_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self.store = store
        self.runner: OsascriptRunner = runner or run_osascript
        self.confirm_timeout_s = confirm_timeout_s
        self.poll_interval_s = poll_interval_s

    async def send(self, to: Any, text: Any) -> dict[str, Any]:
        to_s, text_s = validate_send_args(to, text)
        target: SendTarget = await asyncio.to_thread(self.store.resolve_send_target, to_s)

        baseline: int | None
        try:
            baseline = await asyncio.to_thread(self.store.max_message_rowid)
        except (FileNotFoundError, PermissionError):
            baseline = None

        services = ["chat"] if target.kind == "chat" else ["iMessage", "SMS"]
        used_service: str | None = None
        errors: list[str] = []
        for service in services:
            argv = build_osascript_argv(
                SEND_SCRIPT_LINES,
                [target.kind, target.value, text_s, service if service != "chat" else ""],
            )
            rc, _out, err = await self.runner(argv)
            if rc == 0:
                used_service = None if service == "chat" else service
                break
            lowered = err.lower()
            if any(marker in lowered for marker in _TCC_MARKERS):
                raise PermissionError(automation_hint())
            errors.append(f"{service}: {err.strip()[:300] or f'osascript exit {rc}'}")
        else:
            raise RuntimeError("Messages.app failed to send. " + " | ".join(errors))

        result: dict[str, Any] = {
            "status": "unconfirmed",
            "chat_id": target.chat_id,
            "to": target.value,
            "contact_name": target.contact_name,
            "service": used_service,
        }
        if baseline is None:
            result["note"] = "Sent to Messages.app but chat.db is unreadable, so delivery could not be confirmed."
            return result

        deadline = time.monotonic() + self.confirm_timeout_s
        while True:
            found = await asyncio.to_thread(self.store.find_outgoing, baseline, text_s, target)
            if found is not None:
                result["chat_id"] = found["chat_id"] or result["chat_id"]
                result["message_id"] = found["message_id"]
                if found.get("service"):
                    result["service"] = found["service"]
                if found.get("error"):
                    result["note"] = f"Messages recorded error code {found['error']} for this message."
                else:
                    result["status"] = "sent"
                return result
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(self.poll_interval_s)

        result["note"] = (
            f"Handed to Messages.app but no outgoing row appeared in chat.db within "
            f"{self.confirm_timeout_s:.0f}s; check Messages before retrying to avoid duplicates."
        )
        return result
