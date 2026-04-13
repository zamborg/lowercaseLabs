from __future__ import annotations

import shutil
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.blackhole.storage import AudioStorage
from app.db import Base, engine, ensure_storage_directories
from app.main import app


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    ensure_storage_directories()
    storage = AudioStorage()
    if storage.root.exists():
        shutil.rmtree(storage.root)
    storage.root.mkdir(parents=True, exist_ok=True)

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def auth_headers(client: TestClient, *, display_name: str, email: str) -> dict[str, str]:
    response = client.post("/auth/dev", json={"display_name": display_name, "email": email})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

