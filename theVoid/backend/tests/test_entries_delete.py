from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from app.db import SessionLocal, now_utc
from app.main import app, storage
from app.models import Entry, EntryStatus, Insight, SocialPresence
from app.social import mood_to_dot_color


def _auth(client: TestClient) -> tuple[str, str]:
    identity_token = f"dev-delete-{uuid4().hex}"
    response = client.post("/auth/apple", json={"identity_token": identity_token})
    assert response.status_code == 200
    body = response.json()
    return body["access_token"], body["user"]["id"]


def _make_ready_entry(
    user_id: str,
    local_day: date,
    mood_score: float,
    created_at_offset_seconds: int,
) -> tuple[str, str]:
    entry_id = str(uuid4())
    object_key = f"{user_id}/{entry_id}.m4a"
    entry = Entry(
        id=entry_id,
        user_id=user_id,
        local_date=local_day,
        audio_object_key=object_key,
        duration_seconds=90,
        status=EntryStatus.READY,
        trace_id=str(uuid4()),
        created_at=now_utc() + timedelta(seconds=created_at_offset_seconds),
    )
    insight = Insight(
        entry_id=entry_id,
        mood_score=mood_score,
        mood_tags=["reflection"],
        summary="summary",
        themes=[],
        signals={},
        safety_flags={},
    )
    with SessionLocal() as db:
        db.add(entry)
        db.add(insight)
        db.commit()
    storage.write_bytes(object_key, b"test-audio")
    return entry_id, object_key


def test_delete_entry_removes_entry_audio_and_presence() -> None:
        with TestClient(app) as client:
            access_token, user_id = _auth(client)
            day = date.today()
        entry_id, object_key = _make_ready_entry(
            user_id=user_id,
            local_day=day,
            mood_score=0.7,
            created_at_offset_seconds=0,
        )

        with SessionLocal() as db:
            db.add(
                SocialPresence(
                    user_id=user_id,
                    local_date=day,
                    dot_color=mood_to_dot_color(0.7),
                )
            )
            db.commit()

        response = client.delete(
            f"/entries/{entry_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200

        with SessionLocal() as db:
            assert db.get(Entry, entry_id) is None
            presence = (
                db.query(SocialPresence)
                .filter(SocialPresence.user_id == user_id, SocialPresence.local_date == day)
                .one_or_none()
            )
            assert presence is None

        assert not storage.object_exists(object_key)


def test_delete_latest_entry_recomputes_presence_from_previous_entry() -> None:
        with TestClient(app) as client:
            access_token, user_id = _auth(client)
            day = date.today()

        older_entry_id, older_object_key = _make_ready_entry(
            user_id=user_id,
            local_day=day,
            mood_score=-1.2,
            created_at_offset_seconds=-100,
        )
        newer_entry_id, newer_object_key = _make_ready_entry(
            user_id=user_id,
            local_day=day,
            mood_score=1.4,
            created_at_offset_seconds=0,
        )

        with SessionLocal() as db:
            db.add(
                SocialPresence(
                    user_id=user_id,
                    local_date=day,
                    dot_color=mood_to_dot_color(1.4),
                )
            )
            db.commit()

        response = client.delete(
            f"/entries/{newer_entry_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200

        with SessionLocal() as db:
            assert db.get(Entry, newer_entry_id) is None
            assert db.get(Entry, older_entry_id) is not None
            presence = (
                db.query(SocialPresence)
                .filter(SocialPresence.user_id == user_id, SocialPresence.local_date == day)
                .one_or_none()
            )
            assert presence is not None
            assert presence.dot_color == mood_to_dot_color(-1.2)

        assert not storage.object_exists(newer_object_key)
        storage.delete_object(older_object_key)
