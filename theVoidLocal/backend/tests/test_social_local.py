from __future__ import annotations

from datetime import date
from datetime import timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.social import SILENT_DOT_COLOR


def _auth(client: TestClient, identity_token: str) -> tuple[str, str]:
    response = client.post("/auth/apple", json={"identity_token": identity_token})
    assert response.status_code == 200
    payload = response.json()
    return payload["access_token"], payload["user"]["id"]


def test_publish_local_social_dot_visible_to_friend_feed() -> None:
    with TestClient(app) as client:
        token_a, user_a = _auth(client, f"dev-local-a-{uuid4().hex}")
        token_b, _ = _auth(client, f"dev-local-b-{uuid4().hex}")

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

        local_date = date.today().isoformat()
        publish_response = client.put(
            f"/social/presence/{local_date}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"mood_score": 1.4},
        )
        assert publish_response.status_code == 200

        dots_response = client.get(
            f"/social/dots?local_date={local_date}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]

        target = next((dot for dot in dots if dot["user_id"] == user_a), None)
        assert target is not None
        assert target["has_entry"] is True
        assert target["dot_color"] != SILENT_DOT_COLOR
        assert target["dot_tags"] == []
        assert target["label"]


def test_publish_local_social_dot_with_tags_and_custom_color() -> None:
    with TestClient(app) as client:
        token_a, user_a = _auth(client, f"dev-local-a-tags-{uuid4().hex}")
        token_b, _ = _auth(client, f"dev-local-b-tags-{uuid4().hex}")

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

        local_date = date.today().isoformat()
        publish_response = client.put(
            f"/social/presence/{local_date}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={
                "dot_color": "#66BBAA",
                "mood_tags": ["joyful", "focused", "calm"],
            },
        )
        assert publish_response.status_code == 200

        dots_response = client.get(
            f"/social/dots?local_date={local_date}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]

        target = next((dot for dot in dots if dot["user_id"] == user_a), None)
        assert target is not None
        assert target["has_entry"] is True
        assert target["dot_color"] == "#66BBAA"
        assert target["dot_tags"] == ["joyful", "focused", "calm"]


def test_social_dots_without_date_returns_latest_friend_dot() -> None:
    with TestClient(app) as client:
        token_a, user_a = _auth(client, f"dev-local-a-latest-{uuid4().hex}")
        token_b, _ = _auth(client, f"dev-local-b-latest-{uuid4().hex}")

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

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        today = date.today().isoformat()

        old_publish = client.put(
            f"/social/presence/{yesterday}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={
                "dot_color": "#336699",
                "mood_tags": ["sad"],
            },
        )
        assert old_publish.status_code == 200

        latest_publish = client.put(
            f"/social/presence/{today}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={
                "dot_color": "#88CC44",
                "mood_tags": ["joyful"],
            },
        )
        assert latest_publish.status_code == 200

        dots_response = client.get(
            "/social/dots",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]
        target = next((dot for dot in dots if dot["user_id"] == user_a), None)
        assert target is not None
        assert target["dot_color"] == "#88CC44"
        assert target["dot_tags"] == ["joyful"]


def test_social_dots_history_returns_recent_friend_presences_with_limit() -> None:
    with TestClient(app) as client:
        token_a, user_a = _auth(client, f"dev-local-a-history-{uuid4().hex}")
        token_b, _ = _auth(client, f"dev-local-b-history-{uuid4().hex}")
        token_c, user_c = _auth(client, f"dev-local-c-history-{uuid4().hex}")

        invite_a = client.post(
            "/friends/invite",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"expires_in_days": 7, "max_uses": 1},
        )
        assert invite_a.status_code == 200
        accept_a = client.post(
            "/friends/accept",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"token": invite_a.json()["invite_token"]},
        )
        assert accept_a.status_code == 200

        invite_c = client.post(
            "/friends/invite",
            headers={"Authorization": f"Bearer {token_c}"},
            json={"expires_in_days": 7, "max_uses": 1},
        )
        assert invite_c.status_code == 200
        accept_c = client.post(
            "/friends/accept",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"token": invite_c.json()["invite_token"]},
        )
        assert accept_c.status_code == 200

        oldest = (date.today() - timedelta(days=4)).isoformat()
        middle = (date.today() - timedelta(days=2)).isoformat()
        newest = (date.today() - timedelta(days=1)).isoformat()

        publish_oldest = client.put(
            f"/social/presence/{oldest}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"dot_color": "#3355AA", "mood_tags": ["anxious"]},
        )
        assert publish_oldest.status_code == 200

        publish_middle = client.put(
            f"/social/presence/{middle}/dot",
            headers={"Authorization": f"Bearer {token_c}"},
            json={"dot_color": "#22AA66", "mood_tags": ["calm"]},
        )
        assert publish_middle.status_code == 200

        publish_newest = client.put(
            f"/social/presence/{newest}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"dot_color": "#DD8844", "mood_tags": ["joyful"]},
        )
        assert publish_newest.status_code == 200

        dots_response = client.get(
            "/social/dots?history=true&limit=2",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]

        assert len(dots) == 2
        assert dots[0]["local_date"] == newest
        assert dots[0]["user_id"] == user_a
        assert dots[0]["has_entry"] is True
        assert dots[0]["presence_id"]
        assert dots[1]["local_date"] == middle
        assert dots[1]["user_id"] == user_c
        assert dots[1]["has_entry"] is True
        assert dots[1]["presence_id"]


def test_social_dots_show_friend_label_even_without_entry() -> None:
    with TestClient(app) as client:
        token_a, user_a = _auth(client, f"dev-local-a-label-{uuid4().hex}")
        token_b, _ = _auth(client, f"dev-local-b-label-{uuid4().hex}")

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

        dots_response = client.get("/social/dots", headers={"Authorization": f"Bearer {token_b}"})
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]
        target = next((dot for dot in dots if dot["user_id"] == user_a), None)
        assert target is not None
        assert target["has_entry"] is False
        assert target["dot_color"] == SILENT_DOT_COLOR
        assert target["label"]


def test_delete_social_dot_endpoint_removes_published_dot() -> None:
    with TestClient(app) as client:
        token_a, user_a = _auth(client, f"dev-local-a-delete-{uuid4().hex}")
        token_b, _ = _auth(client, f"dev-local-b-delete-{uuid4().hex}")

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

        local_date = date.today().isoformat()
        publish_response = client.put(
            f"/social/presence/{local_date}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"dot_color": "#88CC44", "mood_tags": ["joyful"]},
        )
        assert publish_response.status_code == 200

        delete_response = client.delete(
            f"/social/presence/{local_date}/dot",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert delete_response.status_code == 200

        dots_response = client.get(
            f"/social/dots?local_date={local_date}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert dots_response.status_code == 200
        dots = dots_response.json()["dots"]
        target = next((dot for dot in dots if dot["user_id"] == user_a), None)
        assert target is not None
        assert target["has_entry"] is False
        assert target["dot_color"] == SILENT_DOT_COLOR
        assert target["dot_tags"] == []
