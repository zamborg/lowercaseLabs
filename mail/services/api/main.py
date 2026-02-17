from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from redis import Redis

from packages.shared.constants import AGENT_QUEUE_NAME, DEFAULT_INBOX_LIMIT, DEFAULT_NOTES_TITLE
from packages.shared.enums import AgentJobType, TriageAction
from packages.shared.schemas import (
    CategoryIn,
    CategoryOut,
    DelegateRequest,
    DraftResponse,
    EmailDetail,
    EmailSummary,
    NotesDocument,
    PolicyIn,
    PolicyOut,
    RememberRequest,
    SnoozeRequest,
)
from services.api import dao
from services.api.auth import require_auth
from services.api.db import Database
from services.api.settings import load_settings
from services.policy.enforcer import policy_allows_send
from services.policy.matcher import suggest_category
from services.policy.validator import validate_policy_payload

app = FastAPI(title="Agentic Mail API", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    settings = load_settings()
    db = Database(settings)
    db.init_schema()
    dev_user = dao.ensure_user(db, settings.dev_user_email)
    dev_account = dao.ensure_account(db, dev_user["id"], settings.dev_user_email)

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)

    app.state.settings = settings
    app.state.db = db
    app.state.dev_user_id = dev_user["id"]
    app.state.dev_account_id = dev_account["id"]
    app.state.redis = redis_client


@app.on_event("shutdown")
def on_shutdown() -> None:
    db: Database = app.state.db
    db.close()


def _loop_status(action: Optional[str]) -> str:
    if action in {TriageAction.ARCHIVE, TriageAction.DELETE}:
        return "closed"
    if action == TriageAction.SNOOZE:
        return "scheduled"
    return "open"


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/inbox", response_model=list[EmailSummary])
def get_inbox(
    request: Request,
    user_id: str = Depends(require_auth),
    view: str = "unread",
    limit: int = DEFAULT_INBOX_LIMIT,
) -> list[dict]:
    db: Database = request.app.state.db
    account_id = request.app.state.dev_account_id
    emails = dao.list_emails(db, account_id, limit, view)
    email_ids = [email["id"] for email in emails]
    triage = dao.get_latest_triage_events(db, user_id, email_ids)
    categories = dao.list_categories(db, user_id)

    results: list[dict] = []
    for email in emails:
        triage_event = triage.get(email["id"])
        loop_status = _loop_status(triage_event["action"] if triage_event else None)

        suggestion = None
        match = suggest_category(email, categories)
        if match:
            suggestion = {
                "id": match.category_id,
                "name": match.category_name,
                "confidence": match.confidence,
            }

        hint = None
        hint_row = dao.get_agent_hint(db, user_id, email["id"])
        if hint_row:
            output = hint_row.get("output") or {}
            hint = {
                "job_id": hint_row["id"],
                "type": hint_row["type"],
                "status": hint_row["status"],
                "summary": output.get("rationale_short"),
            }

        results.append(
            {
                **email,
                "loop_status": loop_status,
                "category_suggestion": suggestion,
                "agent_hint": hint,
            }
        )

    return results


@app.get("/v1/email/{email_id}", response_model=EmailDetail)
def get_email(email_id: str, request: Request, user_id: str = Depends(require_auth)) -> dict:
    db: Database = request.app.state.db
    email = dao.get_email(db, email_id)
    if not email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="email not found")
    context = dao.get_thread_context(db, email.get("thread_key"), email.get("date"), 5)
    email["thread_context"] = context
    return email


def _record_action(
    request: Request,
    user_id: str,
    email_id: str,
    action: str,
    payload: Optional[Dict[str, Any]] = None,
) -> dict:
    db: Database = request.app.state.db
    email = dao.get_email(db, email_id)
    if not email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="email not found")
    dao.insert_triage_event(db, user_id, email_id, action, payload)
    return {"status": "ok"}


@app.post("/v1/email/{email_id}/archive")
def archive_email(email_id: str, request: Request, user_id: str = Depends(require_auth)) -> dict:
    return _record_action(request, user_id, email_id, TriageAction.ARCHIVE)


@app.post("/v1/email/{email_id}/delete")
def delete_email(email_id: str, request: Request, user_id: str = Depends(require_auth)) -> dict:
    return _record_action(request, user_id, email_id, TriageAction.DELETE)


@app.post("/v1/email/{email_id}/snooze")
def snooze_email(
    email_id: str,
    payload: SnoozeRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    return _record_action(
        request,
        user_id,
        email_id,
        TriageAction.SNOOZE,
        {"until": payload.until, "reason": payload.reason},
    )


@app.post("/v1/email/{email_id}/remember")
def remember_email(
    email_id: str,
    payload: RememberRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    db: Database = request.app.state.db
    email = dao.get_email(db, email_id)
    if not email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="email not found")
    dao.insert_triage_event(db, user_id, email_id, TriageAction.REMEMBER, payload.model_dump())

    title = payload.note_category or DEFAULT_NOTES_TITLE
    doc = dao.upsert_notes_document(db, user_id, title)
    summary = payload.why or email.get("snippet") or email.get("subject") or "Remembered email"
    citations = [{"email_id": email_id, "message_id": email.get("message_id")}]
    dao.append_notes_item(db, doc["id"], email_id, summary, [], citations)

    return {"status": "ok", "notes_document_id": doc["id"]}


@app.post("/v1/email/{email_id}/delegate")
def delegate_reply(
    email_id: str,
    payload: DelegateRequest,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    db: Database = request.app.state.db
    email = dao.get_email(db, email_id)
    if not email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="email not found")

    job = dao.create_agent_job(
        db,
        user_id,
        AgentJobType.DRAFT_REPLY,
        {
            "email_id": email_id,
            "category_id": payload.category_id,
            "instruction": payload.instruction,
            "autonomy_override": payload.autonomy_override,
        },
    )

    redis_client: Redis = request.app.state.redis
    redis_client.rpush(AGENT_QUEUE_NAME, job["id"])
    return {"job_id": job["id"]}


@app.get("/v1/drafts/{job_id}", response_model=DraftResponse)
def get_draft(job_id: str, request: Request, user_id: str = Depends(require_auth)) -> dict:
    db: Database = request.app.state.db
    job = dao.get_agent_job(db, job_id)
    if not job or job["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="draft not found")

    output = job.get("output") or {}
    return {
        "job_id": job_id,
        "status": job["status"],
        "draft_text": output.get("primary_artifact"),
        "rationale_short": output.get("rationale_short"),
        "citations": output.get("citations", []) or [],
        "confidence": output.get("confidence"),
        "policy_checks": output.get("policy_checks", []) or [],
    }


@app.post("/v1/drafts/{job_id}/send")
def send_draft(job_id: str, request: Request, user_id: str = Depends(require_auth)) -> JSONResponse:
    db: Database = request.app.state.db
    job = dao.get_agent_job(db, job_id)
    if not job or job["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="draft not found")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="draft not ready")

    output = job.get("output") or {}
    draft_text = output.get("primary_artifact")
    if not draft_text:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="draft missing")

    input_refs = job.get("input_refs") or {}
    email_id = input_refs.get("email_id")
    email = dao.get_email(db, email_id) if email_id else None
    if not email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="email not found")

    policy = None
    category_id = input_refs.get("category_id")
    if category_id:
        category = db.fetch_one("SELECT * FROM categories WHERE id = %s", (category_id,))
        policy = dao.get_policy(db, category.get("default_policy_id") if category else None)

    allowed, reason = policy_allows_send(policy or {}, email.get("subject") or "", draft_text)
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"status": "blocked", "reason": reason},
        )

    dao.insert_triage_event(
        db,
        user_id,
        email_id,
        TriageAction.DELEGATE_REPLY,
        {"job_id": job_id},
    )

    return JSONResponse(content={"status": "sent"})


@app.get("/v1/categories", response_model=list[CategoryOut])
def list_categories(request: Request, user_id: str = Depends(require_auth)) -> list[dict]:
    db: Database = request.app.state.db
    return dao.list_categories(db, user_id)


@app.post("/v1/categories", response_model=CategoryOut)
def create_category(
    payload: CategoryIn,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    db: Database = request.app.state.db
    category = dao.create_category(db, user_id, payload.model_dump())
    return category


@app.put("/v1/categories/{category_id}", response_model=CategoryOut)
def update_category(
    category_id: str,
    payload: CategoryIn,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    db: Database = request.app.state.db
    category = dao.update_category(db, category_id, user_id, payload.model_dump())
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="category not found")
    return category


@app.get("/v1/policies", response_model=list[PolicyOut])
def list_policies(request: Request, user_id: str = Depends(require_auth)) -> list[dict]:
    db: Database = request.app.state.db
    return dao.list_policies(db, user_id)


@app.post("/v1/policies", response_model=PolicyOut)
def create_policy(
    payload: PolicyIn,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    errors = validate_policy_payload(payload.model_dump())
    if errors:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=errors)

    db: Database = request.app.state.db
    policy = dao.create_policy(db, user_id, payload.model_dump())
    return policy


@app.put("/v1/policies/{policy_id}", response_model=PolicyOut)
def update_policy(
    policy_id: str,
    payload: PolicyIn,
    request: Request,
    user_id: str = Depends(require_auth),
) -> dict:
    errors = validate_policy_payload(payload.model_dump())
    if errors:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=errors)

    db: Database = request.app.state.db
    policy = dao.update_policy(db, policy_id, user_id, payload.model_dump())
    if not policy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="policy not found")
    return policy


@app.get("/v1/notes", response_model=list[NotesDocument])
def list_notes(request: Request, user_id: str = Depends(require_auth)) -> list[dict]:
    db: Database = request.app.state.db
    return dao.list_notes_documents(db, user_id)


@app.get("/v1/notes/{doc_id}", response_model=NotesDocument)
def get_notes(doc_id: str, request: Request, user_id: str = Depends(require_auth)) -> dict:
    db: Database = request.app.state.db
    doc = dao.get_notes_document(db, doc_id, user_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="notes not found")
    return doc


@app.post("/v1/notes/rebuild")
def rebuild_notes(payload: Dict[str, str], request: Request, user_id: str = Depends(require_auth)) -> dict:
    scope = payload.get("scope", "all")
    db: Database = request.app.state.db
    job = dao.create_agent_job(db, user_id, AgentJobType.NOTES_UPDATE, {"scope": scope})
    redis_client: Redis = request.app.state.redis
    redis_client.rpush(AGENT_QUEUE_NAME, job["id"])
    return {"job_id": job["id"], "status": "queued"}
