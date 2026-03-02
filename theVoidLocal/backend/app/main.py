from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
from pathlib import Path
import secrets
from threading import Event, Thread
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import and_, func, inspect, text
from sqlalchemy.orm import Session

from .admin import router as admin_router
from .auth import get_current_user, issue_session_for_apple_token
from .config import settings
from .db import Base, engine, ensure_utc, get_db, now_utc
from .jobs import enqueue_job
from .models import (
    Entry,
    EntryStatus,
    FeedbackReport,
    FriendEdge,
    Insight,
    InviteToken,
    Job,
    JobStatus,
    JobType,
    RevealMode,
    SocialPresence,
    Transcript,
    User,
)
from .schemas import (
    AuthAppleRequest,
    AuthSessionResponse,
    CompleteUploadRequest,
    EntryCreateRequest,
    EntryCreateResponse,
    EntryResponse,
    FeedbackCreateRequest,
    FriendAcceptRequest,
    FriendInviteRequest,
    FriendInviteResponse,
    MessageResponse,
    MetricsResponse,
    SocialDot,
    SocialDotsResponse,
    TranscriptPayload,
    InsightPayload,
    UpdateProfileRequest,
    UpdateSocialDotRequest,
    UpdateSocialPresenceRequest,
    UserProfile,
)
from .social import SILENT_DOT_COLOR, mood_to_dot_color, visible_label_for_viewer
from .storage import LocalObjectStorage
from .worker import run_forever as run_worker_forever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("thevoid.api")

app = FastAPI(title=settings.app_name)
storage = LocalObjectStorage(settings.object_storage_root, settings.jwt_secret)
app.include_router(admin_router)
worker_thread: Thread | None = None
worker_stop_event: Event | None = None


def ensure_sqlite_dev_schema_compat() -> None:
    """Patch additive columns for local sqlite dev/test databases."""
    if not str(engine.url).startswith("sqlite"):
        return

    inspector = inspect(engine)
    if "social_presence" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("social_presence")}
    if "dot_tags" not in columns:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE social_presence ADD COLUMN dot_tags JSON NOT NULL DEFAULT '[]'")
            )

if settings.auto_create_schema:
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_dev_schema_compat()


@app.on_event("startup")
def startup() -> None:
    if settings.auto_create_schema:
        Base.metadata.create_all(bind=engine)
        ensure_sqlite_dev_schema_compat()

    global worker_thread, worker_stop_event
    if settings.inline_worker_enabled and worker_thread is None:
        worker_stop_event = Event()
        worker_thread = Thread(
            target=run_worker_forever,
            kwargs={"stop_event": worker_stop_event},
            name="inline-worker",
            daemon=True,
        )
        worker_thread.start()
        logger.info("inline_worker_started")


@app.on_event("shutdown")
def shutdown() -> None:
    global worker_thread, worker_stop_event
    if worker_stop_event is not None:
        worker_stop_event.set()
    worker_thread = None
    worker_stop_event = None


def _model_asset_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (Path(settings.model_assets_root), Path("/app/model_assets")):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        roots.append(candidate)
    return roots


def _normalize_model_asset_path(asset_path: str) -> Path:
    normalized = asset_path.replace("\\", "/").strip("/")
    if not normalized:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found")

    relative_path = Path(normalized)
    if relative_path.is_absolute() or any(part in {"..", "."} for part in relative_path.parts):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found")
    return relative_path


def _resolve_model_asset_file(asset_path: str) -> Path:
    relative_path = _normalize_model_asset_path(asset_path)
    for root in _model_asset_roots():
        candidate = root / relative_path
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root.resolve(strict=False))
        except ValueError:
            continue

        if resolved.is_file():
            return resolved

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "time": now_utc().isoformat()}


@app.get("/models")
def list_models() -> dict:
    files: list[dict] = []
    seen: set[str] = set()
    for root in _model_asset_roots():
        if not root.exists() or not root.is_dir():
            continue

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue

            relative_name = file_path.relative_to(root).as_posix()
            if relative_name in seen:
                continue
            seen.add(relative_name)
            files.append(
                {
                    "name": relative_name,
                    "size_bytes": file_path.stat().st_size,
                }
            )

    files.sort(key=lambda item: item["name"])
    return {"files": files}


@app.get("/models/{asset_path:path}")
def get_model(asset_path: str) -> FileResponse:
    file_path = _resolve_model_asset_file(asset_path)
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@app.post("/auth/apple", response_model=AuthSessionResponse)
def auth_apple(payload: AuthAppleRequest, db: Session = Depends(get_db)) -> AuthSessionResponse:
    user, token = issue_session_for_apple_token(
        db,
        payload.identity_token,
        nonce=payload.nonce,
        display_name=payload.display_name,
        daily_checkin_time_local=payload.daily_checkin_time_local,
        timezone=payload.timezone,
    )
    return AuthSessionResponse(access_token=token, user=UserProfile.model_validate(user))


@app.get("/me", response_model=UserProfile)
def me(current_user: User = Depends(get_current_user)) -> UserProfile:
    return UserProfile.model_validate(current_user)


@app.patch("/me", response_model=UserProfile)
def update_me(
    payload: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserProfile:
    if payload.display_name is not None:
        current_user.display_name = payload.display_name
    if payload.daily_checkin_time_local is not None:
        current_user.daily_checkin_time_local = payload.daily_checkin_time_local
    if payload.timezone is not None:
        current_user.timezone = payload.timezone
    if payload.notification_enabled is not None:
        current_user.notification_enabled = payload.notification_enabled

    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return UserProfile.model_validate(current_user)


@app.post("/feedback", response_model=MessageResponse)
def create_feedback(
    payload: FeedbackCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    normalized_message = payload.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Feedback message cannot be empty")

    report = FeedbackReport(
        user_id=current_user.id,
        kind=payload.kind,
        message=normalized_message,
    )
    db.add(report)
    db.commit()
    return MessageResponse(message="Feedback submitted")


def build_entry_response(entry: Entry) -> EntryResponse:
    transcript_payload = None
    insight_payload = None

    if entry.transcript is not None:
        transcript_payload = TranscriptPayload(
            text=entry.transcript.text,
            provider_metadata=entry.transcript.provider_metadata,
        )

    if entry.insight is not None:
        insight_payload = InsightPayload(
            mood_score=entry.insight.mood_score,
            mood_tags=entry.insight.mood_tags,
            summary=entry.insight.summary,
            themes=entry.insight.themes,
            signals=entry.insight.signals,
            safety_flags=entry.insight.safety_flags,
        )

    return EntryResponse(
        id=entry.id,
        local_date=entry.local_date,
        duration_seconds=entry.duration_seconds,
        status=entry.status,
        created_at=entry.created_at,
        transcript=transcript_payload,
        insight=insight_payload,
    )


def default_friend_label(friend: User) -> str:
    if friend.display_name and friend.display_name.strip():
        return friend.display_name.strip()
    if friend.anonymous_handle and friend.anonymous_handle.strip():
        return friend.anonymous_handle.strip()
    return f"user-{friend.id[:6]}"


@app.post("/entries", response_model=EntryCreateResponse)
def create_entry(
    payload: EntryCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EntryCreateResponse:
    trace_id = str(uuid4())
    entry_id = str(uuid4())
    object_key = storage.create_object_key(current_user.id, entry_id, extension="m4a")

    entry = Entry(
        id=entry_id,
        user_id=current_user.id,
        local_date=payload.local_date,
        duration_seconds=payload.duration_seconds,
        audio_object_key=object_key,
        status=EntryStatus.UPLOADING,
        trace_id=trace_id,
    )
    db.add(entry)
    db.commit()

    upload_token, expires_unix = storage.create_upload_token(current_user.id, object_key)
    upload_base_url = str(request.url_for("upload_object", object_key=object_key))
    upload_url = storage.upload_url(upload_base_url, object_key, upload_token)

    return EntryCreateResponse(
        entry_id=entry_id,
        upload_url=upload_url,
        object_key=object_key,
        expires_at=datetime.fromtimestamp(expires_unix, tz=timezone.utc),
    )


@app.put("/uploads/{object_key:path}", name="upload_object", response_model=MessageResponse)
async def upload_object(
    object_key: str,
    token: str = Query(..., description="Upload token generated by /entries"),
    payload: bytes = Body(..., media_type="application/octet-stream"),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    storage.verify_upload_token(token, current_user.id, object_key)
    storage.write_bytes(object_key, payload)
    return MessageResponse(message=f"uploaded {len(payload)} bytes")


@app.post("/entries/{entry_id}/complete_upload", response_model=MessageResponse)
def complete_upload(
    entry_id: str,
    _: CompleteUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    entry = (
        db.query(Entry)
        .filter(and_(Entry.id == entry_id, Entry.user_id == current_user.id))
        .one_or_none()
    )
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")

    if not storage.object_exists(entry.audio_object_key):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio not uploaded")

    entry.status = EntryStatus.UPLOADED
    db.add(entry)
    db.commit()

    enqueue_job(
        db,
        job_type=JobType.TRANSCRIPTION,
        payload={"entry_id": entry.id, "user_id": current_user.id},
        trace_id=entry.trace_id,
    )

    logger.info(
        "entry_upload_completed",
        extra={"entry_id": entry.id, "user_id": current_user.id, "trace_id": entry.trace_id},
    )

    return MessageResponse(message="upload completed, transcription queued")


@app.get("/entries", response_model=list[EntryResponse])
def list_entries(
    limit: int = Query(default=30, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[EntryResponse]:
    entries = (
        db.query(Entry)
        .filter(Entry.user_id == current_user.id)
        .order_by(Entry.local_date.desc(), Entry.created_at.desc())
        .limit(limit)
        .all()
    )

    for entry in entries:
        entry.transcript = db.get(Transcript, entry.id)
        entry.insight = db.get(Insight, entry.id)

    return [build_entry_response(entry) for entry in entries]


@app.get("/entries/{entry_id}", response_model=EntryResponse)
def get_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EntryResponse:
    entry = (
        db.query(Entry)
        .filter(and_(Entry.id == entry_id, Entry.user_id == current_user.id))
        .one_or_none()
    )
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")

    entry.transcript = db.get(Transcript, entry.id)
    entry.insight = db.get(Insight, entry.id)

    return build_entry_response(entry)


@app.get("/entries/{entry_id}/audio")
def get_entry_audio(
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    entry = (
        db.query(Entry)
        .filter(and_(Entry.id == entry_id, Entry.user_id == current_user.id))
        .one_or_none()
    )
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")

    try:
        audio_bytes = storage.read_bytes(entry.audio_object_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio not found") from exc

    return Response(content=audio_bytes, media_type="audio/m4a")


@app.post("/friends/invite", response_model=FriendInviteResponse)
def create_invite(
    payload: FriendInviteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FriendInviteResponse:
    token = secrets.token_urlsafe(24)
    expires_at = now_utc() + timedelta(days=payload.expires_in_days)

    invite = InviteToken(
        token=token,
        inviter_user_id=current_user.id,
        expires_at=expires_at,
        max_uses=payload.max_uses,
    )
    db.add(invite)
    db.commit()

    return FriendInviteResponse(
        invite_token=token,
        invite_url=f"{settings.invite_base_url}?token={token}",
        expires_at=expires_at,
    )


@app.post("/friends/accept", response_model=MessageResponse)
def accept_invite(
    payload: FriendAcceptRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    invite = db.query(InviteToken).filter(InviteToken.token == payload.token).one_or_none()
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid invite token")

    if ensure_utc(invite.expires_at) < now_utc():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invite expired")

    if invite.use_count >= invite.max_uses:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Invite exhausted")

    if invite.inviter_user_id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot accept your own invite")

    edge = (
        db.query(FriendEdge)
        .filter(and_(FriendEdge.user_id == current_user.id, FriendEdge.friend_user_id == invite.inviter_user_id))
        .one_or_none()
    )
    if edge is None:
        db.add(FriendEdge(user_id=current_user.id, friend_user_id=invite.inviter_user_id))

    reverse = (
        db.query(FriendEdge)
        .filter(and_(FriendEdge.user_id == invite.inviter_user_id, FriendEdge.friend_user_id == current_user.id))
        .one_or_none()
    )
    if reverse is None:
        db.add(FriendEdge(user_id=invite.inviter_user_id, friend_user_id=current_user.id))

    invite.use_count += 1
    db.add(invite)
    db.commit()

    return MessageResponse(message="friend linked")


@app.get("/social/dots", response_model=SocialDotsResponse)
def social_dots(
    local_date: date | None = Query(default=None),
    history: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=300),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SocialDotsResponse:
    day = local_date or date.today()

    edge_pairs = (
        db.query(FriendEdge.user_id, FriendEdge.friend_user_id)
        .filter(or_(FriendEdge.user_id == current_user.id, FriendEdge.friend_user_id == current_user.id))
        .all()
    )
    friend_ids = sorted(
        {
            friend_user_id if user_id == current_user.id else user_id
            for user_id, friend_user_id in edge_pairs
            if user_id != friend_user_id
        }
    )

    if not friend_ids:
        return SocialDotsResponse(local_date=day, dots=[])

    friends = db.query(User).filter(User.id.in_(friend_ids)).all()
    friend_by_id = {friend.id: friend for friend in friends}

    if history and local_date is None:
        presences = (
            db.query(SocialPresence)
            .filter(SocialPresence.user_id.in_(friend_ids))
            .order_by(
                SocialPresence.updated_at.desc(),
                SocialPresence.local_date.desc(),
                SocialPresence.user_id.asc(),
            )
            .limit(limit)
            .all()
        )
        dots: list[SocialDot] = []
        for presence in presences:
            friend = friend_by_id.get(presence.user_id)
            if friend is None:
                continue
            label = visible_label_for_viewer(presence, friend, current_user.id)
            dots.append(
                SocialDot(
                    user_id=presence.user_id,
                    dot_color=presence.dot_color,
                    dot_tags=presence.dot_tags or [],
                    label=label,
                    is_revealed=label is not None,
                    has_entry=True,
                    presence_id=presence.id,
                    local_date=presence.local_date,
                    updated_at=presence.updated_at,
                )
            )

        return SocialDotsResponse(local_date=day, dots=dots)

    if local_date is None:
        presences = (
            db.query(SocialPresence)
            .filter(SocialPresence.user_id.in_(friend_ids))
            .order_by(
                SocialPresence.user_id.asc(),
                SocialPresence.updated_at.desc(),
                SocialPresence.local_date.desc(),
            )
            .all()
        )
        presence_by_user: dict[str, SocialPresence] = {}
        for presence in presences:
            if presence.user_id not in presence_by_user:
                presence_by_user[presence.user_id] = presence
    else:
        presences = (
            db.query(SocialPresence)
            .filter(and_(SocialPresence.user_id.in_(friend_ids), SocialPresence.local_date == day))
            .all()
        )
        presence_by_user = {presence.user_id: presence for presence in presences}

    dots: list[SocialDot] = []
    for friend_id in sorted(friend_ids):
        friend = friend_by_id.get(friend_id)
        if friend is None:
            continue

        presence = presence_by_user.get(friend_id)
        if presence is None:
            label = default_friend_label(friend)
            dots.append(
                SocialDot(
                    user_id=friend_id,
                    dot_color=SILENT_DOT_COLOR,
                    dot_tags=[],
                    label=label,
                    is_revealed=True,
                    has_entry=False,
                )
            )
            continue

        label = visible_label_for_viewer(presence, friend, current_user.id)
        dots.append(
            SocialDot(
                user_id=friend_id,
                dot_color=presence.dot_color,
                dot_tags=presence.dot_tags or [],
                label=label,
                is_revealed=label is not None,
                has_entry=True,
                presence_id=presence.id,
                local_date=presence.local_date,
                updated_at=presence.updated_at,
            )
        )

    return SocialDotsResponse(local_date=day, dots=dots)


@app.patch("/social/presence/{local_date}", response_model=MessageResponse)
def update_social_presence(
    local_date: date,
    payload: UpdateSocialPresenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    if payload.reveal_mode == RevealMode.REVEALED_TO_SPECIFIC and not payload.reveal_friend_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide reveal_friend_ids for selected reveal")

    presence = (
        db.query(SocialPresence)
        .filter(and_(SocialPresence.user_id == current_user.id, SocialPresence.local_date == local_date))
        .one_or_none()
    )
    if presence is None:
        presence = SocialPresence(
            user_id=current_user.id,
            local_date=local_date,
            dot_color=SILENT_DOT_COLOR,
        )

    presence.reveal_mode = payload.reveal_mode
    presence.reveal_friend_ids = sorted(set(payload.reveal_friend_ids))
    presence.display_name_override = payload.display_name_override

    db.add(presence)
    db.commit()

    return MessageResponse(message="social reveal updated")


@app.put("/social/presence/{local_date}/dot", response_model=MessageResponse)
def publish_social_dot(
    local_date: date,
    payload: UpdateSocialDotRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    color = payload.dot_color or mood_to_dot_color(payload.mood_score if payload.mood_score is not None else 0.0)
    tags = [tag.strip().lower() for tag in payload.mood_tags if tag.strip()][:8]

    presence = (
        db.query(SocialPresence)
        .filter(and_(SocialPresence.user_id == current_user.id, SocialPresence.local_date == local_date))
        .one_or_none()
    )
    if presence is None:
        presence = SocialPresence(
            user_id=current_user.id,
            local_date=local_date,
            dot_color=color,
            dot_tags=tags,
        )
    else:
        presence.dot_color = color
        presence.dot_tags = tags

    db.add(presence)
    db.commit()

    return MessageResponse(message="social dot updated")


@app.delete("/social/presence/{local_date}/dot", response_model=MessageResponse)
def delete_social_dot(
    local_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    presence = (
        db.query(SocialPresence)
        .filter(and_(SocialPresence.user_id == current_user.id, SocialPresence.local_date == local_date))
        .one_or_none()
    )
    if presence is None:
        return MessageResponse(message="social dot already absent")

    db.delete(presence)
    db.commit()
    return MessageResponse(message="social dot deleted")


@app.get("/metrics", response_model=MetricsResponse)
def metrics(db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> MetricsResponse:
    today = date.today()
    uploads_today = db.query(func.count(Entry.id)).filter(Entry.local_date == today).scalar() or 0

    one_day_ago = now_utc() - timedelta(hours=24)
    finished_jobs = (
        db.query(Job)
        .filter(and_(Job.finished_at.is_not(None), Job.finished_at >= one_day_ago))
        .all()
    )

    latencies = [
        max((job.finished_at - job.created_at).total_seconds(), 0.0)
        for job in finished_jobs
        if job.finished_at is not None
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    failed_jobs = (
        db.query(func.count(Job.id))
        .filter(and_(Job.status == JobStatus.FAILED, Job.updated_at >= one_day_ago))
        .scalar()
        or 0
    )

    return MetricsResponse(
        uploads_today=uploads_today,
        avg_job_latency_seconds=round(avg_latency, 2),
        failed_jobs_last_24h=failed_jobs,
    )
