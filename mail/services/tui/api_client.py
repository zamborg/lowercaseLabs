import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv


class ApiClient:
    def __init__(self) -> None:
        load_dotenv()
        self._base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        self._token = os.environ.get("DEV_TOKEN", "dev-token")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_inbox(self, view: str = "unread", limit: int = 50) -> list[dict[str, Any]]:
        response = await self._client.get("/v1/inbox", params={"view": view, "limit": limit})
        response.raise_for_status()
        return response.json()

    async def wait_until_ready(self, attempts: int = 10, delay: float = 0.5) -> bool:
        for _ in range(attempts):
            try:
                response = await self._client.get("/v1/health")
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(delay)
        return False

    async def get_email(self, email_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/v1/email/{email_id}")
        response.raise_for_status()
        return response.json()

    async def archive(self, email_id: str) -> None:
        response = await self._client.post(f"/v1/email/{email_id}/archive")
        response.raise_for_status()

    async def delete(self, email_id: str) -> None:
        response = await self._client.post(f"/v1/email/{email_id}/delete")
        response.raise_for_status()

    async def snooze(self, email_id: str, until: str, reason: str | None) -> None:
        response = await self._client.post(
            f"/v1/email/{email_id}/snooze",
            json={"until": until, "reason": reason},
        )
        response.raise_for_status()

    async def remember(self, email_id: str, why: str | None, note_category: str | None) -> None:
        response = await self._client.post(
            f"/v1/email/{email_id}/remember",
            json={"why": why, "note_category": note_category},
        )
        response.raise_for_status()

    async def delegate(self, email_id: str, instruction: str, category_id: str | None) -> str:
        response = await self._client.post(
            f"/v1/email/{email_id}/delegate",
            json={"instruction": instruction, "category_id": category_id},
        )
        response.raise_for_status()
        return response.json()["job_id"]

    async def get_draft(self, job_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/v1/drafts/{job_id}")
        response.raise_for_status()
        return response.json()

    async def send_draft(self, job_id: str) -> dict[str, Any]:
        response = await self._client.post(f"/v1/drafts/{job_id}/send")
        response.raise_for_status()
        return response.json()

    async def list_notes(self) -> list[dict[str, Any]]:
        response = await self._client.get("/v1/notes")
        response.raise_for_status()
        return response.json()

    async def get_notes(self, doc_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/v1/notes/{doc_id}")
        response.raise_for_status()
        return response.json()
