import os
import sys
import json
import time
import hashlib
from typing import Dict, Optional


def get_session_key() -> str:
    """Generate a stable key for the current shell session.

    Prefer the controlling TTY path; fallback to the parent PID when TTY is unavailable.
    The key is a short SHA1 hex digest to be filesystem-safe and compact.
    """
    tty_path: Optional[str]
    try:
        if sys.stdin.isatty():
            tty_path = os.ttyname(0)
        else:
            tty_path = None
    except Exception:
        tty_path = None

    if tty_path:
        basis = f"tty:{tty_path}"
    else:
        basis = f"ppid:{os.getppid()}"

    return hashlib.sha1(basis.encode()).hexdigest()[:16]


def _base_dir() -> str:
    return os.path.expanduser("~/.qork")


def sessions_dir() -> str:
    return os.path.join(_base_dir(), "sessions")


def session_file_path(session_key: str) -> str:
    return os.path.join(sessions_dir(), f"{session_key}.json")

def load_previous_response_id(session_key: str) -> Optional[str]:
    """Return the last stored previous_response_id for this shell session, if any.

    Supports the simplified format:
      { "previous_response_id": "resp_...", "updated_at": "..." }

    If older formats are encountered (dicts without that key), returns None.
    """
    path = session_file_path(session_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            val = data.get("previous_response_id")
            return val if isinstance(val, str) else None
        if isinstance(data, str):
            # In case an older simplistic string-only file was used
            return data
    except Exception:
        return None
    return None


def save_previous_response_id(session_key: str, response_id: str) -> None:
    os.makedirs(sessions_dir(), exist_ok=True)
    payload: Dict[str, str] = {
        "previous_response_id": response_id,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(session_file_path(session_key), "w") as f:
        json.dump(payload, f, indent=2)


