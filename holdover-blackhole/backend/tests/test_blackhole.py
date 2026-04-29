from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import String, func, literal_column, select
from sqlalchemy.dialects import postgresql

from app.blackhole.models import BlackholeCapture, BlackholeNote, BlackholeTodo, BlackholeTranscriptionSession, RoutingTarget
from app.blackhole.services import cleanup_expired_audio
from app.blackhole.storage import AudioStorage
from app.db import SessionLocal
from app.models import utcnow
from tests.conftest import auth_headers


def read_sse_events(client: TestClient, path: str, headers: dict[str, str]) -> dict[str, dict[str, object]]:
    with client.stream("GET", path, headers=headers) as response:
        assert response.status_code == 200
        payload = "".join(chunk for chunk in response.iter_text())

    events: dict[str, dict[str, object]] = {}
    current_event = None
    current_data = None
    for line in payload.splitlines():
        if line.startswith("event: "):
            current_event = line.replace("event: ", "", 1)
        elif line.startswith("data: "):
            import json

            current_data = json.loads(line.replace("data: ", "", 1))
        elif not line.strip() and current_event:
            events[current_event] = current_data or {}
            current_event = None
            current_data = None
    return events


def test_note_crud_enforces_user_ownership(client: TestClient) -> None:
    headers_a = auth_headers(client, display_name="A", email="a@example.com")
    headers_b = auth_headers(client, display_name="B", email="b@example.com")

    created = client.post(
        "/blackhole/notes",
        headers=headers_a,
        json={"title": "Plan", "body_markdown": "Ship it", "tags": ["work"]},
    )
    note_id = created.json()["id"]

    denied = client.get(f"/blackhole/notes/{note_id}", headers=headers_b)
    assert denied.status_code == 404

    patched = client.patch(f"/blackhole/notes/{note_id}", headers=headers_a, json={"archived": True})
    assert patched.status_code == 200

    search = client.get("/blackhole/search", headers=headers_a, params={"q": "Ship"})
    assert search.status_code == 200
    assert search.json()["items"] == []


def test_todo_crud_enforces_user_ownership(client: TestClient) -> None:
    headers_a = auth_headers(client, display_name="A", email="a@example.com")
    headers_b = auth_headers(client, display_name="B", email="b@example.com")

    created = client.post(
        "/blackhole/todos",
        headers=headers_a,
        json={"title": "File taxes", "description_markdown": "Tomorrow", "tags": ["finance"]},
    )
    todo_id = created.json()["id"]

    denied = client.get(f"/blackhole/todos/{todo_id}", headers=headers_b)
    assert denied.status_code == 404

    completed = client.patch(f"/blackhole/todos/{todo_id}", headers=headers_a, json={"status": "completed"})
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["completed_at"] is not None


def test_transcription_sse_and_todo_creation_flow(client: TestClient) -> None:
    headers = auth_headers(client, display_name="A", email="a@example.com")

    response = client.post(
        "/blackhole/transcriptions",
        headers=headers,
        files={"audio": ("capture.m4a", b"remember to file taxes tomorrow")},
        data={"mode": "auto"},
    )
    assert response.status_code == 200
    session_id = response.json()["transcription_session_id"]

    events = read_sse_events(client, f"/blackhole/transcriptions/{session_id}/events", headers)
    assert "uploading" in events
    assert "transcribing" in events
    final_result = events["final_result"]
    assert final_result["route"] == "todo"
    assert final_result["capture_id"] is not None

    create_todo = client.post(
        "/blackhole/todos",
        headers=headers,
        json={
            "title": final_result["route_payload"]["title"],
            "description_markdown": final_result["route_payload"]["description_markdown"],
            "priority": final_result["route_payload"]["priority"],
            "deadline_at": final_result["route_payload"]["deadline_at"],
            "tags": final_result["route_payload"]["tags"],
            "source_capture_id": final_result["capture_id"],
        },
    )
    assert create_todo.status_code == 200
    assert create_todo.json()["title"].lower().startswith("remember")

    with SessionLocal() as db:
        capture = db.scalar(select(BlackholeCapture).where(BlackholeCapture.id == final_result["capture_id"]))
        assert capture is not None
        assert capture.created_record_id == create_todo.json()["id"]


def test_low_confidence_route_does_not_auto_commit_records(client: TestClient) -> None:
    headers = auth_headers(client, display_name="A", email="a@example.com")

    response = client.post(
        "/blackhole/transcriptions",
        headers=headers,
        files={"audio": ("capture.m4a", b"file receipts")},
        data={"mode": "auto"},
    )
    assert response.status_code == 200
    session_id = response.json()["transcription_session_id"]

    session = client.get(f"/blackhole/transcriptions/{session_id}", headers=headers)
    assert session.status_code == 200
    body = session.json()
    assert body["route"] == "todo"
    assert body["route_confidence"] < 0.8

    with SessionLocal() as db:
        assert db.scalar(select(BlackholeNote.id)) is None
        assert db.scalar(select(BlackholeTodo.id)) is None


def test_search_mode_returns_results_without_persisting_capture(client: TestClient) -> None:
    headers = auth_headers(client, display_name="A", email="a@example.com")
    client.post(
        "/blackhole/notes",
        headers=headers,
        json={"title": "Onboarding ideas", "body_markdown": "Voice search should find this note", "tags": ["idea"]},
    )

    response = client.post(
        "/blackhole/transcriptions",
        headers=headers,
        files={"audio": ("capture.m4a", b"find my note about onboarding ideas")},
        data={"mode": "search"},
    )
    session_id = response.json()["transcription_session_id"]
    final_result = read_sse_events(client, f"/blackhole/transcriptions/{session_id}/events", headers)["final_result"]
    assert final_result["route"] == "search"
    assert final_result["capture_id"] is None

    search = client.get("/blackhole/search", headers=headers, params={"q": final_result["route_payload"]["query_text"]})
    assert search.status_code == 200
    assert search.json()["items"][0]["title"] == "Onboarding ideas"

    with SessionLocal() as db:
        assert db.scalar(select(BlackholeCapture.id)) is None


def test_route_failure_defaults_to_note(client: TestClient, monkeypatch) -> None:
    from app.blackhole import services

    headers = auth_headers(client, display_name="A", email="a@example.com")

    class BrokenProvider:
        def transcribe(self, payload: bytes, filename: str | None) -> tuple[str, str]:
            return ("thinking about product direction today", "broken")

        def classify(self, transcript: str, mode: object) -> object:
            raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(services, "get_capture_provider", lambda: BrokenProvider())

    response = client.post(
        "/blackhole/transcriptions",
        headers=headers,
        files={"audio": ("capture.m4a", b"ignored")},
        data={"mode": "auto"},
    )
    session_id = response.json()["transcription_session_id"]
    session = client.get(f"/blackhole/transcriptions/{session_id}", headers=headers)
    body = session.json()

    assert body["status"] == "completed"
    assert body["route"] == "note"
    assert body["route_confidence"] == 0.0


def test_sqlite_search_ranks_title_above_body(client: TestClient) -> None:
    headers = auth_headers(client, display_name="A", email="a@example.com")
    client.post(
        "/blackhole/notes",
        headers=headers,
        json={"title": "Onboarding checklist", "body_markdown": "misc", "tags": []},
    )
    client.post(
        "/blackhole/notes",
        headers=headers,
        json={"title": "General notes", "body_markdown": "Deep onboarding discussion", "tags": []},
    )

    response = client.get("/blackhole/search", headers=headers, params={"q": "onboarding"})
    titles = [item["title"] for item in response.json()["items"]]
    assert titles[0] == "Onboarding checklist"


def test_postgres_search_vector_uses_tsvector_concat_operator() -> None:
    tags_text = func.coalesce(func.cast(BlackholeNote.tags, String), "")
    title_vector = func.setweight(func.to_tsvector("english", func.coalesce(BlackholeNote.title, "")), literal_column("'A'"))
    tags_vector = func.setweight(func.to_tsvector("english", tags_text), literal_column("'B'"))
    body_vector = func.setweight(func.to_tsvector("english", func.coalesce(BlackholeNote.body_markdown, "")), literal_column("'C'"))
    vector = title_vector.op("||")(tags_vector).op("||")(body_vector)
    ts_query = func.plainto_tsquery("english", "onboarding")
    stmt = select(func.ts_rank(vector, ts_query)).where(vector.op("@@")(ts_query))

    compiled = str(stmt.compile(dialect=postgresql.dialect()))
    assert " || " in compiled
    assert " + " not in compiled
    assert "setweight" in compiled
    assert "'A'" in compiled
    assert "setweight_1" not in compiled


def test_audio_cleanup_removes_expired_objects() -> None:
    storage = AudioStorage()
    with SessionLocal() as db:
        from app.models import User

        user = User(display_name="Cleaner", email="cleaner@example.com", auth_provider="dev")
        db.add(user)
        db.commit()
        db.refresh(user)

        object_key = storage.save_bytes(user_id=user.id, filename="expired.m4a", payload=b"expired")
        capture = BlackholeCapture(
            user_id=user.id,
            mode="note",
            audio_object_key=object_key,
            transcript_text="expired",
            transcript_provider="stub",
            routing_target=RoutingTarget.note,
            routing_confidence=1.0,
            routing_payload={},
            created_at=utcnow() - timedelta(hours=48),
            updated_at=utcnow() - timedelta(hours=48),
        )
        db.add(capture)
        db.commit()

        deleted = cleanup_expired_audio(db)
        assert deleted == 1
        assert not storage.path_for(object_key).exists()
