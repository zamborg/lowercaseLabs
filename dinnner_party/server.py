#!/usr/bin/env python3

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
PEOPLE_DIR = ROOT / "people"
CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs"
RUNTIME_DIR = ROOT / ".runtime"
TASK_TEMPLATE = ROOT / "task.md"
HOST = "127.0.0.1"
PORT = 8000
DEFAULT_RUN_MODEL = "gpt-5.1"


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, cwd=str(ROOT), check=False)


def load_calendar_snapshot() -> dict[str, Any]:
    return json.loads((DATA_DIR / "calendar.json").read_text(encoding="utf-8"))


def load_calendar_connector() -> dict[str, Any]:
    return json.loads((CONFIG_DIR / "calendar-connector.json").read_text(encoding="utf-8"))


def load_people() -> dict[str, dict[str, Any]]:
    PEOPLE_DIR.mkdir(exist_ok=True)
    people: dict[str, dict[str, Any]] = {}
    for directory in sorted(PEOPLE_DIR.iterdir()):
        if not directory.is_dir():
            continue
        person_path = directory / "person.json"
        soul_path = directory / "soul.md"
        if not person_path.exists() or not soul_path.exists():
            continue
        person = json.loads(person_path.read_text(encoding="utf-8"))
        person["soul"] = soul_path.read_text(encoding="utf-8")
        person["directory"] = str(directory)
        person["soul_path"] = str(soul_path)
        people[person["id"]] = person
    return people


def save_person(payload: dict[str, Any]) -> dict[str, Any]:
    display_name = str(payload.get("display_name", "")).strip()
    requested_id = str(payload.get("id", "")).strip()
    person_id = slugify(requested_id or display_name)
    soul = str(payload.get("soul", "")).strip()
    calendar_identity = str(payload.get("calendar_identity", "")).strip()
    requested_mcp = str(payload.get("calendar_mcp_server", "")).strip()
    calendar_mcp_server = requested_mcp or f"google-calendar-{person_id}"
    preferred_model = str(payload.get("preferred_model", DEFAULT_RUN_MODEL)).strip() or DEFAULT_RUN_MODEL

    if not person_id:
        raise ValueError("display_name or id is required")
    if not display_name:
        raise ValueError("display_name is required")
    if not soul:
        raise ValueError("soul is required")

    directory = PEOPLE_DIR / person_id
    directory.mkdir(parents=True, exist_ok=True)

    person = {
        "id": person_id,
        "display_name": display_name,
        "preferred_model": preferred_model,
        "calendar_identity": calendar_identity,
        "calendar_mcp_server": calendar_mcp_server,
        "role": "participant",
    }
    (directory / "person.json").write_text(json.dumps(person, indent=2) + "\n", encoding="utf-8")
    (directory / "soul.md").write_text(soul + "\n", encoding="utf-8")
    return load_people()[person_id]


def delete_person(person_id: str) -> None:
    directory = PEOPLE_DIR / person_id
    if not directory.exists():
        raise FileNotFoundError(person_id)
    shutil.rmtree(directory)


CALENDAR_SNAPSHOT = load_calendar_snapshot()


@dataclass
class Message:
    id: int
    sender: str
    text: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "text": self.text,
            "created_at": self.created_at,
        }


class MCPManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def connect(self, name: str, url: str, scopes: str = "") -> dict[str, Any]:
        with self._lock:
            self._reap_locked()
            existing = self._sessions.get(name)
            if existing is not None and existing.get("running"):
                return self.status_locked([name])[name]

            record = {
                "name": name,
                "url": url,
                "scopes": scopes,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "running": True,
                "phase": "configuring",
                "output": "",
                "auth_url": None,
                "exit_code": None,
                "error": None,
                "process": None,
            }
            self._sessions[name] = record

            thread = threading.Thread(target=self._setup_and_login, args=(name, url, scopes), daemon=True)
            thread.start()
            record["thread"] = thread

            return self.status_locked([name])[name]

    def cancel(self, name: str) -> dict[str, Any]:
        with self._lock:
            record = self._sessions.get(name)
            process = record.get("process") if record else None
            if process is not None and process.poll() is None:
                process.kill()
                record["running"] = False
                record["phase"] = "cancelled"
            self._reap_locked()
            return self.status_locked([name])[name]

    def status(self, names: list[str]) -> dict[str, dict[str, Any]]:
        with self._lock:
            self._reap_locked()
            return self.status_locked(names)

    def _ensure_server_locked(self, name: str, url: str) -> None:
        current = self._get_server_details(name)
        if current is None:
            add = run_cmd(["codex", "mcp", "add", name, "--url", url])
            if add.returncode != 0:
                raise RuntimeError(add.stderr.strip() or add.stdout.strip() or "failed to add mcp server")
            return

        current_url = current.get("url", "")
        if current_url == url:
            return

        remove = run_cmd(["codex", "mcp", "remove", name])
        if remove.returncode != 0:
            raise RuntimeError(remove.stderr.strip() or remove.stdout.strip() or "failed to remove existing mcp server")

        add = run_cmd(["codex", "mcp", "add", name, "--url", url])
        if add.returncode != 0:
            raise RuntimeError(add.stderr.strip() or add.stdout.strip() or "failed to add mcp server")

    def _get_server_details(self, name: str) -> dict[str, str] | None:
        result = run_cmd(["codex", "mcp", "get", name])
        if result.returncode != 0:
            return None

        details: dict[str, str] = {"name": name}
        for line in result.stdout.splitlines()[1:]:
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            details[key.strip()] = value.strip()
        return details

    def _list_rows(self) -> dict[str, dict[str, str]]:
        result = run_cmd(["codex", "mcp", "list"])
        rows: dict[str, dict[str, str]] = {}
        if result.returncode != 0:
            return rows

        for line in result.stdout.splitlines()[1:]:
            stripped = line.rstrip()
            if not stripped:
                continue
            parts = re.split(r"\s{2,}", stripped.strip())
            if len(parts) < 5:
                continue
            rows[parts[0]] = {
                "name": parts[0],
                "url": parts[1],
                "status": parts[-2],
                "auth": parts[-1],
            }
        return rows

    def _append_output(self, name: str, text: str) -> None:
        with self._lock:
            current = self._sessions.get(name)
            if current is None:
                return
            current["output"] += text
            if current["auth_url"] is None:
                match = re.search(r"https?://\S+", text)
                if match:
                    current["auth_url"] = match.group(0)
                    current["phase"] = "awaiting_browser"

    def _setup_and_login(self, name: str, url: str, scopes: str) -> None:
        try:
            self._ensure_server_locked(name, url)
            with self._lock:
                current = self._sessions.get(name)
                if current is None:
                    return
                current["phase"] = "logging_in"

            command = ["codex", "mcp", "login", name]
            if scopes.strip():
                command.extend(["--scopes", scopes.strip()])

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                cwd=str(ROOT),
            )
            with self._lock:
                current = self._sessions.get(name)
                if current is None:
                    process.kill()
                    return
                current["process"] = process

            assert process.stdout is not None
            for line in process.stdout:
                self._append_output(name, line)

            process.wait()
            with self._lock:
                current = self._sessions.get(name)
                if current is not None:
                    current["exit_code"] = process.returncode
                    current["running"] = False
                    current["phase"] = "completed" if process.returncode == 0 else "failed"
        except Exception as exc:
            with self._lock:
                current = self._sessions.get(name)
                if current is not None:
                    current["running"] = False
                    current["phase"] = "failed"
                    current["error"] = str(exc)

    def _reap_locked(self) -> None:
        for record in self._sessions.values():
            process = record.get("process")
            if process is None:
                continue
            exit_code = process.poll()
            if exit_code is not None:
                record["exit_code"] = exit_code
                record["running"] = False

    def status_locked(self, names: list[str]) -> dict[str, dict[str, Any]]:
        rows = self._list_rows()
        statuses: dict[str, dict[str, Any]] = {}

        for name in sorted(set(names)):
            config = self._get_server_details(name)
            row = rows.get(name, {})
            session = self._sessions.get(name)
            statuses[name] = {
                "name": name,
                "configured": config is not None,
                "transport": config.get("transport") if config else None,
                "url": config.get("url") if config else None,
                "enabled": row.get("status"),
                "auth": row.get("auth"),
                "login_session": {
                    "running": session["running"],
                    "started_at": session["started_at"],
                    "phase": session["phase"],
                    "auth_url": session["auth_url"],
                    "exit_code": session["exit_code"],
                    "error": session["error"],
                    "output_tail": session["output"][-1200:],
                } if session else None,
            }
        return statuses


class ChatState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: list[Message] = []
        self._next_id = 1
        self.reset()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._messages = []
            self._next_id = 1
            self._append_locked("system", "Room reset. Dinner planning is open.")
            return self.snapshot_locked()

    def _append_locked(self, sender: str, text: str) -> Message:
        message = Message(
            id=self._next_id,
            sender=sender,
            text=text,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self._messages.append(message)
        self._next_id += 1
        return message

    def send_message(self, sender: str, text: str) -> dict[str, Any]:
        clean_sender = sender.strip()
        clean_text = text.strip()
        if not clean_sender or not clean_text:
            raise ValueError("sender and text are required")

        with self._lock:
            return self._append_locked(clean_sender, clean_text).to_dict()

    def read_messages(self, after: int = 0) -> dict[str, Any]:
        with self._lock:
            unread = [message.to_dict() for message in self._messages if message.id > after]
            return {
                "messages": unread,
                "last_id": self._messages[-1].id if self._messages else 0,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict[str, Any]:
        messages = [message.to_dict() for message in self._messages]
        return {
            "messages": messages,
            "last_id": messages[-1]["id"] if messages else 0,
            "agents": sorted(load_people().keys()),
        }


class AgentRunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._current_run_id: str | None = None
        LOG_DIR.mkdir(exist_ok=True)
        RUNTIME_DIR.mkdir(exist_ok=True)

    def start(self, model: str = DEFAULT_RUN_MODEL, reset_room: bool = True) -> dict[str, Any]:
        with self._lock:
            people = load_people()
            self._reap_locked()
            if self._has_running_locked():
                raise RuntimeError("agents are already running")
            if not people:
                raise RuntimeError("register at least one agent before running")

            if reset_room:
                CHAT_STATE.reset()

            self._records = {}
            self._current_run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")

            try:
                for person_id in sorted(people.keys()):
                    person = people[person_id]
                    prompt_text = self._build_prompt(person, people, model)
                    prompt_path = RUNTIME_DIR / f"{self._current_run_id}-{person_id}.prompt.md"
                    prompt_path.write_text(prompt_text, encoding="utf-8")

                    log_path = LOG_DIR / f"{person_id}.log"
                    log_handle = log_path.open("w", encoding="utf-8")

                    process = subprocess.Popen(
                        [
                            "codex",
                            "exec",
                            "--skip-git-repo-check",
                            "--dangerously-bypass-approvals-and-sandbox",
                            "-m",
                            model,
                            "-C",
                            str(ROOT),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=str(ROOT),
                    )
                    assert process.stdin is not None
                    process.stdin.write(prompt_text)
                    process.stdin.close()

                    self._records[person_id] = {
                        "agent": person_id,
                        "display_name": person["display_name"],
                        "model": model,
                        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "pid": process.pid,
                        "log_path": str(log_path),
                        "prompt_path": str(prompt_path),
                        "process": process,
                        "log_handle": log_handle,
                    }
            except OSError as exc:
                self._kill_all_locked()
                self._records = {}
                self._current_run_id = None
                raise RuntimeError(f"failed to launch codex exec: {exc}") from exc

            return self.status_locked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._kill_all_locked()
            self._reap_locked()
            return self.status_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reap_locked()
            return self.status_locked()

    def _build_prompt(
        self,
        person: dict[str, Any],
        people: dict[str, dict[str, Any]],
        model: str,
    ) -> str:
        task = TASK_TEMPLATE.read_text(encoding="utf-8")
        other_names = [other["display_name"] for other_id, other in people.items() if other_id != person["id"]]
        participants = ", ".join(other_names) if other_names else "no one else yet"

        prompt = (
            task.replace("__AGENT__", person["id"])
            .replace("__MODEL__", model)
            .replace("__SOUL_PATH__", person["soul_path"])
            .replace("__PARTICIPANTS__", participants)
        )

        person_structured = {key: value for key, value in person.items() if key not in {"soul", "directory", "soul_path"}}
        return (
            f"{prompt}\n\n"
            "# Person Record\n\n"
            "```json\n"
            f"{json.dumps(person_structured, indent=2)}\n"
            "```\n\n"
            "# Soul\n\n"
            f"{person['soul']}\n"
        )

    def _reap_locked(self) -> None:
        finished: list[str] = []
        for person_id, record in self._records.items():
            process = record["process"]
            exit_code = process.poll()
            if exit_code is None:
                continue
            record["exit_code"] = exit_code
            log_handle = record.get("log_handle")
            if log_handle is not None and not log_handle.closed:
                log_handle.close()
            finished.append(person_id)

        for person_id in finished:
            self._records[person_id].pop("log_handle", None)

    def _kill_all_locked(self) -> None:
        for record in self._records.values():
            process = record["process"]
            if process.poll() is None:
                process.kill()
            log_handle = record.get("log_handle")
            if log_handle is not None and not log_handle.closed:
                log_handle.close()

    def _has_running_locked(self) -> bool:
        return any(record["process"].poll() is None for record in self._records.values())

    def status_locked(self) -> dict[str, Any]:
        current_people = load_people()
        all_ids = sorted(set(current_people.keys()) | set(self._records.keys()))
        agents: list[dict[str, Any]] = []
        any_running = False

        for person_id in all_ids:
            record = self._records.get(person_id)
            person = current_people.get(person_id)
            if record is None:
                agents.append(
                    {
                        "agent": person_id,
                        "display_name": person["display_name"] if person else person_id,
                        "running": False,
                        "pid": None,
                        "model": person["preferred_model"] if person else DEFAULT_RUN_MODEL,
                        "log_path": str(LOG_DIR / f"{person_id}.log"),
                        "prompt_path": None,
                        "exit_code": None,
                    }
                )
                continue

            process = record["process"]
            exit_code = process.poll()
            running = exit_code is None
            any_running = any_running or running
            agents.append(
                {
                    "agent": person_id,
                    "display_name": record["display_name"],
                    "running": running,
                    "pid": record["pid"],
                    "model": record["model"],
                    "log_path": record["log_path"],
                    "prompt_path": record["prompt_path"],
                    "started_at": record["started_at"],
                    "exit_code": exit_code,
                }
            )

        return {
            "run_id": self._current_run_id,
            "running": any_running,
            "agents": agents,
        }


CHAT_STATE = ChatState()
MCP_MANAGER = MCPManager()
AGENT_RUNNER = AgentRunManager()


HTTP_SPEC = {
    "name": "Dinner Party Chat API",
    "base_url": f"http://{HOST}:{PORT}",
    "notes": [
        "Agents are file-backed under /people with person.json plus soul.md.",
        "Each person should use a unique Google Calendar MCP server name, for example google-calendar-zubin.",
        "The UI can start Codex MCP login, open the OAuth URL, and poll completion.",
    ],
    "primitives": [
        {"name": "read_messages", "method": "GET", "path": "/api/read_messages?after=<last_seen_id>"},
        {"name": "send_message", "method": "POST", "path": "/api/send_message"},
        {"name": "people", "method": "POST", "path": "/api/people"},
        {"name": "mcp_connect", "method": "POST", "path": "/api/mcp/connect"},
        {"name": "mcp_status", "method": "GET", "path": "/api/mcp/status?name=<server_name>"},
        {"name": "start_agents", "method": "POST", "path": "/api/agent_runs/start"},
    ],
}


class DinnerPartyHandler(BaseHTTPRequestHandler):
    server_version = "DinnerPartyHTTP/0.4"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers("application/json")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return

        if path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            self._serve_file(STATIC_DIR / relative)
            return

        if path == "/api/spec":
            self._send_json(HTTP_SPEC)
            return

        if path == "/api/state":
            self._send_json(self._build_state_payload())
            return

        if path == "/api/read_messages":
            after = self._coerce_int(query.get("after", ["0"])[0], default=0)
            self._send_json(CHAT_STATE.read_messages(after=after))
            return

        if path == "/api/calendar":
            self._send_json(CALENDAR_SNAPSHOT)
            return

        if path == "/api/calendar_connector":
            self._send_json(load_calendar_connector())
            return

        if path == "/api/people":
            people = load_people()
            self._send_json(
                {
                    "people": {
                        person_id: self._serialize_person(person)
                        for person_id, person in sorted(people.items())
                    }
                }
            )
            return

        if path.startswith("/api/people/"):
            person_id = path.removeprefix("/api/people/")
            people = load_people()
            if person_id not in people:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown person: {person_id}")
                return
            self._send_json(self._serialize_person(people[person_id]))
            return

        if path == "/api/mcp/status":
            requested = query.get("name", [])
            names = requested if requested else self._current_mcp_names()
            self._send_json({"mcp_statuses": MCP_MANAGER.status(names)})
            return

        if path == "/api/agent_runs/status":
            self._send_json(AGENT_RUNNER.status())
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json_body()

        if path == "/api/send_message":
            try:
                message = CHAT_STATE.send_message(
                    sender=str(payload.get("sender", "")),
                    text=str(payload.get("text", "")),
                )
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json({"message": message}, status=HTTPStatus.CREATED)
            return

        if path == "/api/reset":
            self._send_json({"state": CHAT_STATE.reset()})
            return

        if path == "/api/people":
            try:
                person = save_person(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json({"person": self._serialize_person(person)}, status=HTTPStatus.CREATED)
            return

        if path == "/api/mcp/connect":
            name = str(payload.get("name", "")).strip()
            url = str(payload.get("url", "")).strip()
            scopes = str(payload.get("scopes", "")).strip()
            if not name or not url:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "name and url are required")
                return
            try:
                status = MCP_MANAGER.connect(name=name, url=url, scopes=scopes)
            except RuntimeError as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            self._send_json({"mcp_status": status}, status=HTTPStatus.ACCEPTED)
            return

        if path == "/api/mcp/cancel":
            name = str(payload.get("name", "")).strip()
            if not name:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "name is required")
                return
            self._send_json({"mcp_status": MCP_MANAGER.cancel(name)})
            return

        if path == "/api/agent_runs/start":
            model = str(payload.get("model", DEFAULT_RUN_MODEL)).strip() or DEFAULT_RUN_MODEL
            reset_room = bool(payload.get("reset_room", True))
            try:
                status = AGENT_RUNNER.start(model=model, reset_room=reset_room)
            except RuntimeError as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            self._send_json(status, status=HTTPStatus.ACCEPTED)
            return

        if path == "/api/agent_runs/stop":
            self._send_json(AGENT_RUNNER.stop())
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/people/"):
            person_id = path.removeprefix("/api/people/")
            try:
                delete_person(person_id)
            except FileNotFoundError:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown person: {person_id}")
                return
            self._send_json({"deleted": person_id})
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _build_state_payload(self) -> dict[str, Any]:
        people = load_people()
        mcp_names = [person["calendar_mcp_server"] for person in people.values()]
        state = CHAT_STATE.snapshot()
        state["calendar"] = CALENDAR_SNAPSHOT
        state["calendar_connector"] = load_calendar_connector()
        state["people"] = {
            person_id: self._serialize_person(person) for person_id, person in sorted(people.items())
        }
        state["mcp_statuses"] = MCP_MANAGER.status(mcp_names)
        state["agent_run"] = AGENT_RUNNER.status()
        state["defaults"] = {
            "run_model": DEFAULT_RUN_MODEL,
            "calendar_mcp_server_prefix": "google-calendar-",
        }
        return state

    def _serialize_person(self, person: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": person["id"],
            "display_name": person["display_name"],
            "preferred_model": person["preferred_model"],
            "calendar_identity": person["calendar_identity"],
            "calendar_mcp_server": person["calendar_mcp_server"],
            "role": person["role"],
            "directory": person["directory"],
            "soul_path": person["soul_path"],
            "soul": person["soul"],
        }

    def _current_mcp_names(self) -> list[str]:
        return [person["calendar_mcp_server"] for person in load_people().values()]

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_common_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_common_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": int(status)}, status=status)

    def _serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "file not found")
            return

        if STATIC_DIR not in resolved.parents and resolved != STATIC_DIR / "index.html":
            self._send_error_json(HTTPStatus.FORBIDDEN, "forbidden")
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_common_headers(content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _coerce_int(value: str, default: int) -> int:
        try:
            return int(value)
        except ValueError:
            return default


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DinnerPartyHandler)
    print(f"Dinner Party server running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
