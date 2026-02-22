from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import FeedbackReport


def _auth(client: TestClient, identity_token: str) -> tuple[str, str]:
    response = client.post("/auth/apple", json={"identity_token": identity_token})
    assert response.status_code == 200
    payload = response.json()
    return payload["access_token"], payload["user"]["id"]


def test_feedback_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json={"kind": "idea", "message": "This is an unauthenticated request."},
        )
    assert response.status_code == 401


def test_feedback_submission_persists_report() -> None:
    with TestClient(app) as client:
        token, user_id = _auth(client, f"dev-feedback-{uuid4().hex}")
        message = "Could we add weekly summary trends to the journal?"

        response = client.post(
            "/feedback",
            headers={"Authorization": f"Bearer {token}"},
            json={"kind": "idea", "message": message},
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Feedback submitted"

    with SessionLocal() as db:
        report = (
            db.query(FeedbackReport)
            .filter(
                FeedbackReport.user_id == user_id,
                FeedbackReport.kind == "idea",
                FeedbackReport.message == message,
            )
            .order_by(FeedbackReport.id.desc())
            .first()
        )
        assert report is not None
