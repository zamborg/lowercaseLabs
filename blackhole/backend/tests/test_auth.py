from fastapi.testclient import TestClient

from tests.conftest import auth_headers


def test_dev_auth_and_me(client: TestClient) -> None:
    headers = auth_headers(client, display_name="Zubin", email="zubin@example.com")
    response = client.get("/me", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Zubin"
    assert body["auth_provider"] == "dev"

