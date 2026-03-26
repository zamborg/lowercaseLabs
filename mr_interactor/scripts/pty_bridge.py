#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import pty
import signal
import struct
import subprocess
import sys
import termios
import threading
from typing import Any


def send(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def set_winsize(fd: int, rows: int, cols: int) -> None:
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    if hasattr(termios, "tcsetwinsize"):
        try:
            termios.tcsetwinsize(fd, (rows, cols))
            return
        except Exception:
            pass
    # Some Python builds lack tcsetwinsize on pty master, so keep ioctl-compatible bytes around.
    try:
        import fcntl

        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided to pty bridge")

    master_fd, slave_fd = pty.openpty()
    set_winsize(master_fd, args.rows, args.cols)
    proc = subprocess.Popen(
        command,
        cwd=args.cwd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    stop_event = threading.Event()

    def read_loop() -> None:
        while not stop_event.is_set():
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            send({"type": "data", "data": base64.b64encode(data).decode("ascii")})

    reader = threading.Thread(target=read_loop, daemon=True)
    reader.start()

    try:
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            message = json.loads(raw_line)
            event_type = message.get("type")

            if event_type == "write":
                payload = base64.b64decode(message["data"])
                os.write(master_fd, payload)
            elif event_type == "resize":
                rows = int(message["rows"])
                cols = int(message["cols"])
                set_winsize(master_fd, rows, cols)
                if proc.poll() is None:
                    proc.send_signal(signal.SIGWINCH)
            elif event_type == "stop":
                if proc.poll() is None:
                    proc.terminate()
                break
    finally:
        stop_event.set()
        try:
            os.close(master_fd)
        except OSError:
            pass

    code = proc.wait()
    send({"type": "exit", "code": code})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
