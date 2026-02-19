from datetime import date
from uuid import uuid4

from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import AccountDecommission


def test_admin_requires_basic_auth() -> None:
    with TestClient(app) as client:
        response = client.get("/admin")
    assert response.status_code == 401


def test_admin_allows_valid_credentials() -> None:
    with TestClient(app) as client:
        response = client.get("/admin", auth=("admin", "admin"))
    assert response.status_code == 200
    assert "theVoid Admin" in response.text


def test_admin_users_page_renders() -> None:
    with TestClient(app) as client:
        response = client.get("/admin/users", auth=("admin", "admin"))
    assert response.status_code == 200
    assert "Account Lifecycle" in response.text


def test_admin_dots_page_renders() -> None:
    with TestClient(app) as client:
        response = client.get("/admin/dots", auth=("admin", "admin"))
    assert response.status_code == 200
    assert "Social Dots" in response.text


def test_admin_dots_page_shows_published_dot() -> None:
    identity_token = f"dev-dots-{uuid4().hex}"
    local_date = date.today().isoformat()

    with TestClient(app) as client:
        auth_response = client.post("/auth/apple", json={"identity_token": identity_token})
        assert auth_response.status_code == 200
        body = auth_response.json()
        access_token = body["access_token"]

        publish_response = client.put(
            f"/social/presence/{local_date}/dot",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "dot_color": "#66BBAA",
                "mood_tags": ["joyful", "calm"],
            },
        )
        assert publish_response.status_code == 200

        dots_response = client.get("/admin/dots", auth=("admin", "admin"))
        assert dots_response.status_code == 200
        assert "#66BBAA" in dots_response.text
        assert "joyful, calm" in dots_response.text


def test_admin_overview_shows_recommission_controls() -> None:
    identity_token = f"dev-admin-overview-{uuid4().hex}"

    with TestClient(app) as client:
        auth_response = client.post("/auth/apple", json={"identity_token": identity_token})
        assert auth_response.status_code == 200
        user_id = auth_response.json()["user"]["id"]

        decommission_response = client.post(
            f"/admin/users/{user_id}/decommission",
            data={"reason": "dashboard test", "redirect_to": "/admin"},
            auth=("admin", "admin"),
            follow_redirects=False,
        )
        assert decommission_response.status_code == 303

        overview = client.get("/admin", auth=("admin", "admin"))
        assert overview.status_code == 200
        assert "Decommissioned Accounts" in overview.text
        assert "Recommission" in overview.text
        assert user_id[:8] in overview.text


def test_admin_can_decommission_and_recommission_user() -> None:
    identity_token = f"dev-lifecycle-{uuid4().hex}"
    with TestClient(app) as client:
        auth_response = client.post("/auth/apple", json={"identity_token": identity_token})
        assert auth_response.status_code == 200
        body = auth_response.json()
        access_token = body["access_token"]
        user_id = body["user"]["id"]

        decommission_response = client.post(
            f"/admin/users/{user_id}/decommission",
            data={"reason": "testing lifecycle", "redirect_to": "/admin/users"},
            auth=("admin", "admin"),
            follow_redirects=False,
        )
        assert decommission_response.status_code == 303

        with SessionLocal() as db:
            decommission = (
                db.query(AccountDecommission)
                .filter(AccountDecommission.user_id == user_id)
                .one_or_none()
            )
            assert decommission is not None
            assert decommission.reason == "testing lifecycle"

        me_response = client.get("/me", headers={"Authorization": f"Bearer {access_token}"})
        assert me_response.status_code == 403

        second_auth_response = client.post("/auth/apple", json={"identity_token": identity_token})
        assert second_auth_response.status_code == 403

        recommission_response = client.post(
            f"/admin/users/{user_id}/recommission",
            data={"redirect_to": "/admin/users"},
            auth=("admin", "admin"),
            follow_redirects=False,
        )
        assert recommission_response.status_code == 303

        with SessionLocal() as db:
            decommission = (
                db.query(AccountDecommission)
                .filter(AccountDecommission.user_id == user_id)
                .one_or_none()
            )
            assert decommission is None

        recovered_auth = client.post("/auth/apple", json={"identity_token": identity_token})
        assert recovered_auth.status_code == 200
