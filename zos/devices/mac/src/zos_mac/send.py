"""Send iMessage/SMS via macOS Shortcuts (primary) or Messages.app AppleScript, then confirm via chat.db.

- Primary backend (`shortcuts`): writes `{"to": handle, "text": text}` to a private `0600`
  temp file (removed in `finally`) and runs `/usr/bin/shortcuts run "ZOS Send Message" --input-path <file>`
  via `create_subprocess_exec` (no shell, argv only).
- Fallback / group-chat backend (`applescript`): passes recipient and text strictly as
  `osascript` argv items (`on run argv`), never interpolated into script source.
- Never logs message text or recipients.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from zos_mac.chatdb import ChatStore, SendTarget

MAX_SEND_CHARS = 4000
DEFAULT_CONFIRM_TIMEOUT_S = 5.0
DEFAULT_POLL_INTERVAL_S = 0.5
OSASCRIPT_TIMEOUT_S = 30.0
SHORTCUTS_TIMEOUT_S = 30.0
SHORTCUTS_LIST_TIMEOUT_S = 10.0
SHORTCUTS_CACHE_TTL_S = 60.0

SHORTCUTS_BIN = "/usr/bin/shortcuts"
SHORTCUT_NAME = "ZOS Send Message"
VALID_SEND_BACKENDS = ("auto", "shortcuts", "applescript")

SubprocessRunner = Callable[[Sequence[str]], Awaitable[tuple[int, str, str]]]
OsascriptRunner = SubprocessRunner

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
_SHORTCUT_MISSING_MARKERS = ("couldn't find shortcut", "could not find shortcut", "not found", "doesn't exist")
_SHORTCUT_PRIVACY_MARKERS = ("not allowed", "denied", "permission", "privacy", "not permitted")


def validate_send_backend(value: str) -> str:
    clean = str(value or "auto").strip().lower()
    if clean not in VALID_SEND_BACKENDS:
        raise ValueError(
            f"send_backend must be one of {', '.join(VALID_SEND_BACKENDS)}; got {value!r}"
        )
    return clean


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


def build_shortcuts_list_argv() -> list[str]:
    return [SHORTCUTS_BIN, "list"]


def build_shortcuts_run_argv(input_path: str, *, shortcut_name: str = SHORTCUT_NAME) -> list[str]:
    return [SHORTCUTS_BIN, "run", shortcut_name, "--input-path", input_path]


async def _run_subprocess(
    argv: Sequence[str], *, timeout_s: float, label: str
) -> tuple[int, str, str]:
    """Execute a subprocess without a shell, stdin DEVNULL, and a hard timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return 127, "", f"{label} executable not found: {exc.filename or argv[0]}"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return 124, "", f"{label} timed out"
    return (
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


async def run_osascript(argv: Sequence[str]) -> tuple[int, str, str]:
    """Default AppleScript runner: execute osascript without a shell, with a hard timeout."""
    return await _run_subprocess(argv, timeout_s=OSASCRIPT_TIMEOUT_S, label="osascript")


async def run_shortcuts(argv: Sequence[str]) -> tuple[int, str, str]:
    """Default Shortcuts CLI runner: execute /usr/bin/shortcuts without a shell, with a hard timeout."""
    is_list = len(argv) >= 2 and argv[1] == "list"
    timeout_s = SHORTCUTS_LIST_TIMEOUT_S if is_list else SHORTCUTS_TIMEOUT_S
    return await _run_subprocess(argv, timeout_s=timeout_s, label="shortcuts")


def automation_hint() -> str:
    return (
        "macOS blocked sending (Automation permission). Allow it in System Settings -> "
        "Privacy & Security -> Automation: enable 'Messages' for "
        f"{os.path.realpath(sys.executable)}, or install the '{SHORTCUT_NAME}' Shortcut "
        "(see devices/mac/README.md)."
    )


def osascript_timeout_hint() -> str:
    return (
        "osascript timed out: macOS is likely waiting on the Automation -> Messages prompt "
        "on the Mac; consider the Shortcuts backend (see devices/mac/README.md)."
    )


def shortcut_not_installed_hint(shortcut_name: str = SHORTCUT_NAME) -> str:
    return (
        f"Shortcut '{shortcut_name}' is not installed on this Mac. "
        f"Import devices/mac/shortcuts/{shortcut_name}.shortcut (or build it manually) — "
        "see devices/mac/README.md for setup instructions."
    )


def _sanitize_stderr(stderr: str, *, redact: Sequence[str] = (), max_chars: int = 240) -> str:
    cleaned = stderr or ""
    for secret in redact:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


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
    """Resolves a recipient, sends via Shortcuts or Messages.app AppleScript, and confirms via chat.db."""

    def __init__(
        self,
        store: ChatStore,
        *,
        send_backend: str = "auto",
        runner: SubprocessRunner | None = None,
        shortcuts_runner: SubprocessRunner | None = None,
        confirm_timeout_s: float = DEFAULT_CONFIRM_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        shortcuts_cache_ttl_s: float = SHORTCUTS_CACHE_TTL_S,
        shortcut_name: str = SHORTCUT_NAME,
    ) -> None:
        self.store = store
        self.send_backend = validate_send_backend(send_backend)
        self.runner: SubprocessRunner = runner or run_osascript
        self.shortcuts_runner: SubprocessRunner = shortcuts_runner or run_shortcuts
        self.confirm_timeout_s = confirm_timeout_s
        self.poll_interval_s = poll_interval_s
        self.shortcuts_cache_ttl_s = shortcuts_cache_ttl_s
        self.shortcut_name = shortcut_name
        self._shortcut_cache: tuple[float, bool] | None = None

    def clear_shortcut_cache(self) -> None:
        self._shortcut_cache = None

    async def has_shortcut(self) -> bool:
        """Return True if `self.shortcut_name` appears in `shortcuts list` (briefly cached)."""
        now = time.monotonic()
        if self._shortcut_cache is not None:
            checked_at, present = self._shortcut_cache
            if (now - checked_at) < self.shortcuts_cache_ttl_s:
                return present

        rc, out, _err = await self.shortcuts_runner(build_shortcuts_list_argv())
        present = rc == 0 and any(line.strip() == self.shortcut_name for line in out.splitlines())
        self._shortcut_cache = (time.monotonic(), present)
        return present

    async def _send_via_shortcuts(self, handle: str, text: str) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix="zos-send-", suffix=".json")
        try:
            os.fchmod(fd, 0o600)
            payload = json.dumps({"to": handle, "text": text}, ensure_ascii=False).encode("utf-8")
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            fd = -1  # closed by fdopen

            argv = build_shortcuts_run_argv(tmp_path, shortcut_name=self.shortcut_name)
            rc, _out, err = await self.shortcuts_runner(argv)
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

        if rc == 0:
            return
        if rc == 124 or "timed out" in err.lower():
            raise RuntimeError(
                f"Shortcuts timed out running '{self.shortcut_name}'; check whether macOS is "
                "waiting on a Shortcuts 'Allow / Always Allow' privacy prompt on the Mac."
            )
        lowered = err.lower()
        if any(marker in lowered for marker in _SHORTCUT_MISSING_MARKERS):
            self.clear_shortcut_cache()
            raise RuntimeError(shortcut_not_installed_hint(self.shortcut_name))
        summary = _sanitize_stderr(err, redact=(handle, text)) or f"exit {rc}"
        if any(marker in lowered for marker in _SHORTCUT_PRIVACY_MARKERS):
            raise PermissionError(
                f"Shortcuts denied sending via '{self.shortcut_name}' ({summary}). "
                "Allow Messages access for the Shortcut (choose Always Allow when prompted)."
            )
        raise RuntimeError(f"Shortcuts failed to send via '{self.shortcut_name}' (exit {rc}): {summary}")

    async def _send_via_applescript(self, target: SendTarget, text: str) -> str | None:
        services = ["chat"] if target.kind == "chat" else ["iMessage", "SMS"]
        errors: list[str] = []
        for service in services:
            argv = build_osascript_argv(
                SEND_SCRIPT_LINES,
                [target.kind, target.value, text, service if service != "chat" else ""],
            )
            rc, _out, err = await self.runner(argv)
            if rc == 0:
                return None if service == "chat" else service
            lowered = err.lower()
            if rc == 124 or "timed out" in lowered:
                raise RuntimeError(osascript_timeout_hint())
            if any(marker in lowered for marker in _TCC_MARKERS):
                raise PermissionError(automation_hint())
            summary = _sanitize_stderr(err, redact=(target.value, text)) or f"osascript exit {rc}"
            errors.append(f"{service}: {summary}")
        raise RuntimeError("Messages.app failed to send. " + " | ".join(errors))

    async def send(self, to: Any, text: Any) -> dict[str, Any]:
        to_s, text_s = validate_send_args(to, text)
        target: SendTarget = await asyncio.to_thread(self.store.resolve_send_target, to_s)

        baseline: int | None
        try:
            baseline = await asyncio.to_thread(self.store.max_message_rowid)
        except (FileNotFoundError, PermissionError):
            baseline = None

        used_backend: str
        used_service: str | None = None

        if target.kind == "chat":
            # Shortcuts' Send Message cannot address a chat by id (group chats).
            if self.send_backend == "shortcuts":
                raise RuntimeError(
                    "Group chats and chat_id targets cannot be sent via Shortcuts; "
                    'group sends need the AppleScript backend (`send_backend = "auto"` or '
                    '"applescript") and the macOS Automation -> Messages grant.'
                )
            used_backend = "applescript"
            used_service = await self._send_via_applescript(target, text_s)
        elif self.send_backend == "shortcuts":
            if not await self.has_shortcut():
                raise RuntimeError(shortcut_not_installed_hint(self.shortcut_name))
            used_backend = "shortcuts"
            await self._send_via_shortcuts(target.value, text_s)
        elif self.send_backend == "applescript":
            used_backend = "applescript"
            used_service = await self._send_via_applescript(target, text_s)
        else:
            # auto: prefer Shortcuts when "ZOS Send Message" is installed, else AppleScript
            if await self.has_shortcut():
                used_backend = "shortcuts"
                await self._send_via_shortcuts(target.value, text_s)
            else:
                used_backend = "applescript"
                used_service = await self._send_via_applescript(target, text_s)

        result: dict[str, Any] = {
            "status": "unconfirmed",
            "backend": used_backend,
            "chat_id": target.chat_id,
            "to": target.value,
            "contact_name": target.contact_name,
            "service": used_service,
        }
        if baseline is None:
            result["note"] = (
                f"Sent via {used_backend} but chat.db is unreadable, so delivery could not be confirmed."
            )
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
            f"Handed to Messages ({used_backend}) but no outgoing row appeared in chat.db within "
            f"{self.confirm_timeout_s:.0f}s; check Messages before retrying to avoid duplicates."
        )
        return result
