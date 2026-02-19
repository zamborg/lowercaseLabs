from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _auth(client: TestClient, token_suffix: str, display_name: str | None = None) -> dict:
    payload = {"identity_token": f"dev-{token_suffix}-{uuid4().hex}"}
    if display_name is not None:
        payload["display_name"] = display_name
    response = client.post("/auth/apple", json=payload)
    assert response.status_code == 200
    return response.json()


def test_social_dots_returns_display_name_label() -> None:
    with TestClient(app) as client:
        user_a = _auth(client, "social-a", display_name="Alice")
        user_b = _auth(client, "social-b", display_name="Bob")

        invite_response = client.post(
            "/friends/invite",
            json={},
            headers={"Authorization": f"Bearer {user_a['access_token']}"},
        )
        assert invite_response.status_code == 200
        invite_token = invite_response.json()["invite_token"]

        accept_response = client.post(
            "/friends/accept",
            json={"token": invite_token},
            headers={"Authorization": f"Bearer {user_b['access_token']}"},
        )
        assert accept_response.status_code == 200

        dots_response = client.get(
            "/social/dots",
            headers={"Authorization": f"Bearer {user_b['access_token']}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]
        assert any(dot["label"] == "Alice" for dot in dots)


def test_social_dots_label_falls_back_to_handle_when_display_name_missing() -> None:
    with TestClient(app) as client:
        user_a = _auth(client, "social-handle-a")
        user_b = _auth(client, "social-handle-b", display_name="Bob")

        invite_response = client.post(
            "/friends/invite",
            json={},
            headers={"Authorization": f"Bearer {user_a['access_token']}"},
        )
        assert invite_response.status_code == 200
        invite_token = invite_response.json()["invite_token"]

        accept_response = client.post(
            "/friends/accept",
            json={"token": invite_token},
            headers={"Authorization": f"Bearer {user_b['access_token']}"},
        )
        assert accept_response.status_code == 200

        dots_response = client.get(
            "/social/dots",
            headers={"Authorization": f"Bearer {user_b['access_token']}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]
        expected_handle = user_a["user"]["anonymous_handle"]
        assert any(dot["label"] == expected_handle for dot in dots)
