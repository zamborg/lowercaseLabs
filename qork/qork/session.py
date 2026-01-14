from __future__ import annotations

import os
from typing import Optional


def _base_dir() -> str:
    return os.path.expanduser("~/.qork")


def history_dir() -> str:
    return os.path.join(_base_dir(), "history")


def thread_file_path() -> str:
    return os.path.join(history_dir(), "session.id")


def load_thread_response_id() -> Optional[str]:
    path = thread_file_path()
    try:
        with open(path, "r") as f:
            val = f.read().strip()
        return val or None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def save_thread_response_id(response_id: str) -> None:
    os.makedirs(history_dir(), exist_ok=True)
    with open(thread_file_path(), "w") as f:
        f.write(response_id.strip() + "\n")

