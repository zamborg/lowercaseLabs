"""Persistent local tmux shell runner (`tmux -L zos`) for `mac_shell_run`."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S = 300
MAX_OUTPUT_BYTES = 32 * 1024  # 32 KB head + tail cap
IDLE_SHELL_COMMANDS = frozenset(
    {"zsh", "bash", "sh", "fish", "dash", "ksh", "-zsh", "-bash", "-sh", "-fish"}
)


def cap_output_bytes(raw: bytes, *, max_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Cap output at max_bytes using head + tail preservation."""
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace"), False
    half = max_bytes // 2
    omitted = len(raw) - (half * 2)
    head = raw[:half].decode("utf-8", errors="replace")
    tail = raw[-half:].decode("utf-8", errors="replace")
    return f"{head}\n...[truncated {omitted} bytes]...\n{tail}", True


class TmuxShellRunner:
    """Runs commands sequentially in a dedicated `tmux -L <socket>` persistent shell pane."""

    def __init__(
        self,
        *,
        socket_name: str = "zos",
        session_name: str = "zos",
        temp_root: Path | None = None,
    ) -> None:
        self.socket_name = socket_name
        self.session_name = session_name
        self.temp_root = temp_root or Path("/tmp/zos")
        self._lock = asyncio.Lock()

    @property
    def target_pane(self) -> str:
        return f"{self.session_name}:0.0"

    async def _tmux(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "tmux",
            "-L",
            self.socket_name,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", errors="replace").strip(),
            stderr_b.decode("utf-8", errors="replace").strip(),
        )

    async def _has_session(self) -> bool:
        rc, _, _ = await self._tmux("has-session", "-t", self.session_name)
        return rc == 0

    async def _ensure_session(self) -> bool:
        """Ensure the dedicated tmux session exists. Returns True if newly created."""
        if await self._has_session():
            return False
        rc, _, stderr = await self._tmux(
            "new-session",
            "-d",
            "-s",
            self.session_name,
            "-x",
            "200",
            "-y",
            "50",
        )
        if rc != 0:
            raise RuntimeError(f"Failed to start tmux session '{self.session_name}': {stderr}")
        # Wait briefly for the shell to initialize inside the new pane
        for _ in range(30):
            cmd = await self._pane_current_command()
            if cmd in IDLE_SHELL_COMMANDS:
                break
            await asyncio.sleep(0.05)
        return True

    async def _pane_current_command(self) -> str | None:
        rc, out, _ = await self._tmux(
            "display-message",
            "-p",
            "-t",
            self.target_pane,
            "#{pane_current_command}",
        )
        if rc != 0 or not out:
            return None
        return out.strip()

    async def run(
        self,
        command: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Run `command` via per-call temp script `source` in the persistent tmux shell."""
        if not command or not command.strip():
            raise ValueError("command must be a non-empty string")
        clamped_timeout = max(1.0, min(float(timeout_s), float(MAX_TIMEOUT_S)))

        async with self._lock:
            await self._ensure_session()

            current_cmd = await self._pane_current_command()
            if current_cmd is None:
                await self._ensure_session()
                current_cmd = await self._pane_current_command()

            if current_cmd not in IDLE_SHELL_COMMANDS:
                if not reset:
                    return {
                        "ok": False,
                        "error_code": "PANE_BUSY",
                        "current_command": current_cmd,
                        "message": (
                            f"PANE_BUSY: shell pane is currently running '{current_cmd}'. "
                            "Wait for it to finish or call with reset=true to send Ctrl-C."
                        ),
                    }
                # Send Ctrl-C and wait up to 3s for shell to return
                await self._tmux("send-keys", "-t", self.target_pane, "C-c")
                deadline_reset = time.monotonic() + 3.0
                while time.monotonic() < deadline_reset:
                    await asyncio.sleep(0.1)
                    current_cmd = await self._pane_current_command()
                    if current_cmd is None:
                        await self._ensure_session()
                        current_cmd = await self._pane_current_command()
                    if current_cmd in IDLE_SHELL_COMMANDS:
                        break
                if current_cmd not in IDLE_SHELL_COMMANDS:
                    return {
                        "ok": False,
                        "error_code": "PANE_BUSY",
                        "current_command": current_cmd,
                        "message": (
                            f"PANE_BUSY: pane did not return to shell after Ctrl-C "
                            f"(still running '{current_cmd}')."
                        ),
                    }

            self.temp_root.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self.temp_root, 0o700)

            call_id = uuid.uuid4().hex[:12]
            sh_path = self.temp_root / f"{call_id}.sh"
            out_path = self.temp_root / f"{call_id}.out"
            rc_path = self.temp_root / f"{call_id}.rc"

            sh_path.write_text(command if command.endswith("\n") else command + "\n", encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(sh_path, 0o600)

            # Only send a fixed safe `source` invocation — avoids quoting, multiline, and `!` hazards
            # while keeping cwd and exported env vars persistent in the shell pane.
            invocation = f"source {sh_path} >{out_path} 2>&1; echo $? >{rc_path}"
            rc_send, _, stderr_send = await self._tmux(
                "send-keys", "-t", self.target_pane, "-l", invocation
            )
            if rc_send != 0:
                self._cleanup_files(sh_path, out_path, rc_path)
                raise RuntimeError(f"tmux send-keys failed: {stderr_send}")
            await self._tmux("send-keys", "-t", self.target_pane, "Enter")

            deadline = time.monotonic() + clamped_timeout
            pane_died = False
            exit_code: int | None = None

            while time.monotonic() < deadline:
                if rc_path.exists():
                    rc_text = rc_path.read_text(encoding="utf-8", errors="replace").strip()
                    if rc_text:
                        try:
                            exit_code = int(rc_text.splitlines()[0].strip())
                        except ValueError:
                            exit_code = -1
                        break
                if not await self._has_session():
                    pane_died = True
                    break
                await asyncio.sleep(0.05)

            raw_out = out_path.read_bytes() if out_path.exists() else b""
            output, truncated = cap_output_bytes(raw_out)

            if pane_died:
                await self._ensure_session()
                self._cleanup_files(sh_path, out_path, rc_path)
                return {
                    "ok": True,
                    "exit_code": None,
                    "output": output,
                    "truncated": truncated,
                    "timed_out": False,
                    "pane_recreated": True,
                    "note": "Shell pane exited during command execution and was recreated.",
                }

            if exit_code is None:
                # Timed out; leave script/output readable no longer needed after reading
                self._cleanup_files(sh_path, out_path, rc_path)
                return {
                    "ok": False,
                    "error_code": "TIMEOUT",
                    "exit_code": None,
                    "output": output,
                    "truncated": truncated,
                    "timed_out": True,
                    "pane_recreated": False,
                    "note": (
                        f"Command timed out after {clamped_timeout:.0f}s. "
                        "Pass reset=true on the next call to interrupt with Ctrl-C."
                    ),
                }

            self._cleanup_files(sh_path, out_path, rc_path)
            return {
                "ok": True,
                "exit_code": exit_code,
                "output": output,
                "truncated": truncated,
                "timed_out": False,
                "pane_recreated": False,
            }

    @staticmethod
    def _cleanup_files(*paths: Path) -> None:
        for p in paths:
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
