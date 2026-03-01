from datetime import date, timedelta
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


def test_admin_feedback_page_renders() -> None:
    with TestClient(app) as client:
        response = client.get("/admin/feedback", auth=("admin", "admin"))
    assert response.status_code == 200
    assert "Feedback Reports" in response.text


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


def test_admin_dots_page_defaults_to_all_events_ordered_by_updates() -> None:
    identity_token = f"dev-dots-order-{uuid4().hex}"
    today = date.today()
    older_day = (today - timedelta(days=1)).isoformat()
    newer_day = today.isoformat()
    older_color = f"#{uuid4().hex[:6].upper()}"
    newer_color = f"#{uuid4().hex[:6].upper()}"

    with TestClient(app) as client:
        auth_response = client.post("/auth/apple", json={"identity_token": identity_token})
        assert auth_response.status_code == 200
        body = auth_response.json()
        access_token = body["access_token"]
        user_id = body["user"]["id"]

        older_response = client.put(
            f"/social/presence/{older_day}/dot",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"dot_color": older_color, "mood_tags": ["calm"]},
        )
        assert older_response.status_code == 200

        newer_response = client.put(
            f"/social/presence/{newer_day}/dot",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"dot_color": newer_color, "mood_tags": ["focused"]},
        )
        assert newer_response.status_code == 200

        page_response = client.get("/admin/dots", auth=("admin", "admin"))
        assert page_response.status_code == 200
        html = page_response.text

        assert "All events (ordered)" in html
        assert html.count(f"<td>{user_id[:8]}</td>") >= 2
        assert newer_color in html
        assert older_color in html
        assert html.index(newer_color) < html.index(older_color)


def test_admin_dots_page_shows_undirected_friend_graph() -> None:
    identity_token_a = f"dev-friend-graph-a-{uuid4().hex}"
    identity_token_b = f"dev-friend-graph-b-{uuid4().hex}"

    with TestClient(app) as client:
        auth_a = client.post("/auth/apple", json={"identity_token": identity_token_a})
        assert auth_a.status_code == 200
        body_a = auth_a.json()
        token_a = body_a["access_token"]
        user_a = body_a["user"]["id"]
        handle_a = body_a["user"]["anonymous_handle"]

        auth_b = client.post("/auth/apple", json={"identity_token": identity_token_b})
        assert auth_b.status_code == 200
        body_b = auth_b.json()
        token_b = body_b["access_token"]
        user_b = body_b["user"]["id"]
        handle_b = body_b["user"]["anonymous_handle"]

        invite_response = client.post(
            "/friends/invite",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"expires_in_days": 7, "max_uses": 1},
        )
        assert invite_response.status_code == 200
        invite_token = invite_response.json()["invite_token"]

        accept_response = client.post(
            "/friends/accept",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"token": invite_token},
        )
        assert accept_response.status_code == 200

        page_response = client.get("/admin/dots", auth=("admin", "admin"))
        assert page_response.status_code == 200
        html = page_response.text

        left_id, right_id = sorted([user_a[:8], user_b[:8]])
        assert "Friend Graph" in html
        assert "Undirected edges represent reciprocal friendships between users." in html
        assert "Friend Pairs" in html
        assert "aria-label=\"Undirected friend graph\"" in html
        assert 'if (char === \'"\') return "&quot;";' in html
        assert f"@{handle_a}" in html
        assert f"@{handle_b}" in html
        assert f"<td>{left_id}</td>" in html
        assert f"<td>{right_id}</td>" in html


def test_admin_feedback_page_shows_submitted_feedback() -> None:
    identity_token = f"dev-feedback-admin-{uuid4().hex}"

    with TestClient(app) as client:
        auth_response = client.post("/auth/apple", json={"identity_token": identity_token})
        assert auth_response.status_code == 200
        access_token = auth_response.json()["access_token"]

        feedback_response = client.post(
            "/feedback",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "kind": "bug",
                "message": "Settings screen clipped when keyboard opens.",
            },
        )
        assert feedback_response.status_code == 200

        page_response = client.get("/admin/feedback", auth=("admin", "admin"))
        assert page_response.status_code == 200
        assert "Settings screen clipped when keyboard opens." in page_response.text
        assert "bug" in page_response.text


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
