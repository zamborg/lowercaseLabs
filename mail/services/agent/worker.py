from __future__ import annotations

import time
from datetime import datetime, timezone
from redis import Redis

from packages.shared.constants import AGENT_QUEUE_NAME, DEFAULT_NOTES_TITLE
from packages.shared.enums import AgentJobStatus, AgentJobType
from services.agent import dao
from services.agent.db import Database
from services.agent.logic import build_notes_markdown, classify_email, generate_draft, suggest_snooze
from services.agent.settings import load_settings


def _add_provenance(output: dict, job: dict, email_ids: list[str]) -> dict:
    output["provenance"] = {
        "model": "stub",
        "prompt_hash": "na",
        "tool_calls": [],
        "input_email_ids": email_ids,
        "job_id": job.get("id"),
        "job_type": job.get("type"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return output


def _process_job(db: Database, job_id: str, signature: str) -> None:
    job = dao.get_agent_job(db, job_id)
    if not job:
        return

    dao.update_agent_job(db, job_id, AgentJobStatus.RUNNING, None, None)

    job_type = job.get("type")
    input_refs = job.get("input_refs") or {}

    try:
        if job_type == AgentJobType.DRAFT_REPLY:
            email_id = input_refs.get("email_id")
            instruction = input_refs.get("instruction", "")
            email = dao.get_email(db, email_id)
            if not email:
                raise RuntimeError("email not found")
            category = dao.get_category(db, input_refs.get("category_id"))
            policy = dao.get_policy(db, category.get("default_policy_id") if category else None)
            output = generate_draft(email, instruction, policy, signature)
            output = _add_provenance(output, job, [email_id])
            dao.update_agent_job(db, job_id, AgentJobStatus.SUCCEEDED, output, None)
            return

        if job_type == AgentJobType.SNOOZE_SUGGEST:
            email_id = input_refs.get("email_id")
            email = dao.get_email(db, email_id)
            if not email:
                raise RuntimeError("email not found")
            output = suggest_snooze(email)
            output = _add_provenance(output, job, [email_id])
            dao.update_agent_job(db, job_id, AgentJobStatus.SUCCEEDED, output, None)
            return

        if job_type == AgentJobType.CLASSIFY:
            email_id = input_refs.get("email_id")
            email = dao.get_email(db, email_id)
            if not email:
                raise RuntimeError("email not found")
            categories = dao.list_categories(db, job["user_id"])
            output = classify_email(email, categories)
            output = _add_provenance(output, job, [email_id])
            dao.update_agent_job(db, job_id, AgentJobStatus.SUCCEEDED, output, None)
            return

        if job_type == AgentJobType.NOTES_UPDATE:
            scope = input_refs.get("scope", "all")
            title = input_refs.get("title") or (DEFAULT_NOTES_TITLE if scope == "all" else f"Notes {scope}")
            doc = dao.upsert_notes_document(db, job["user_id"], title)
            items = dao.list_notes_items(db, doc["id"], limit=100)
            content = build_notes_markdown(items)
            dao.update_notes_document_content(db, doc["id"], content)
            output = {
                "primary_artifact": {"notes_document_id": doc["id"]},
                "rationale_short": "Rebuilt notes document from recent items.",
                "citations": [item.get("email_id") for item in items],
                "confidence": 0.6,
                "policy_checks": [],
            }
            output = _add_provenance(output, job, [item.get("email_id") for item in items])
            dao.update_agent_job(db, job_id, AgentJobStatus.SUCCEEDED, output, None)
            return

        raise RuntimeError(f"unknown job type: {job_type}")
    except Exception as exc:
        dao.update_agent_job(db, job_id, AgentJobStatus.FAILED, None, str(exc))


def run() -> None:
    settings = load_settings()
    db = Database(settings)
    db.init_schema()

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    signature = settings.default_signature

    while True:
        item = redis_client.blpop(AGENT_QUEUE_NAME, timeout=5)
        if not item:
            time.sleep(settings.poll_interval)
            continue
        _, job_id = item
        _process_job(db, job_id, signature)


if __name__ == "__main__":
    run()
