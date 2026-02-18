from fastapi.testclient import TestClient

from app.main import app


def test_admin_requires_basic_auth() -> None:
    with TestClient(app) as client:
        response = client.get("/admin")
    assert response.status_code == 401


def test_admin_allows_valid_credentials() -> None:
    with TestClient(app) as client:
        response = client.get("/admin", auth=("admin", "admin"))
    assert response.status_code == 200
    assert "theVoid Admin" in response.text


def test_admin_transcripts_page_renders() -> None:
    with TestClient(app) as client:
        response = client.get("/admin/transcripts", auth=("admin", "admin"))
    assert response.status_code == 200
    assert "Transcripts" in response.text
