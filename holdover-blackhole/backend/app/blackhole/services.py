from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from fastapi import BackgroundTasks, HTTPException, status
from openai import OpenAI
from sqlalchemy import String, case, desc, func, literal_column, or_, select
from sqlalchemy.orm import Session

from app.blackhole.models import (
    BlackholeCapture,
    BlackholeNote,
    BlackholeTodo,
    BlackholeTranscriptionSession,
    CaptureMode,
    CaptureRecordMode,
    RoutingTarget,
    TodoPriority,
    TodoStatus,
    TranscriptionSessionStatus,
)
from app.blackhole.schemas import (
    NoteCreate,
    NoteDraftPayload,
    NotePage,
    NoteRead,
    NoteUpdate,
    SearchDraftPayload,
    SearchPage,
    SearchResult,
    TodoCreate,
    TodoDraftPayload,
    TodoPage,
    TodoRead,
    TodoUpdate,
    TranscriptionFinalResult,
)
from app.config import get_settings
from app.db import SessionLocal
from app.metrics import (
    low_confidence_total,
    note_creation_total,
    route_distribution_total,
    search_latency_seconds,
    todo_creation_total,
    transcription_latency_seconds,
    transcription_sessions_total,
    zero_result_search_total,
)
from app.models import AuditMessage, User, utcnow
from app.blackhole.storage import AudioStorage

logger = logging.getLogger(__name__)


def encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("utf-8")).decode("utf-8")


def decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        return int(base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8"))
    except Exception:
        return 0


def clamp_limit(value: int | None) -> int:
    settings = get_settings()
    raw = value or settings.blackhole_search_default_limit
    return max(1, min(raw, settings.blackhole_search_max_limit))


def log_audit(db: Session, category: str, message: str) -> None:
    db.add(AuditMessage(category=category, message=message))
    db.commit()


def extract_title(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return fallback
    words = cleaned.split(" ")
    return " ".join(words[:8]).strip().capitalize()


def extract_tags(text: str) -> list[str]:
    lowered = text.lower()
    candidates = []
    for word in ("work", "personal", "idea", "planning", "urgent", "followup", "search"):
        if word in lowered:
            candidates.append(word)
    return candidates[:4]


def infer_deadline(text: str) -> datetime | None:
    lowered = text.lower()
    now = utcnow()
    if "tomorrow" in lowered:
        return now + timedelta(days=1)
    if "next week" in lowered:
        return now + timedelta(days=7)
    return None


def normalize_search_query(text: str) -> str:
    text = re.sub(r"^(find|search for|search|look up|look for)\s+", "", text.strip(), flags=re.IGNORECASE)
    return text.strip() or text.strip()


@dataclass
class ProviderResult:
    transcript: str
    provider: str
    route: RoutingTarget
    route_confidence: float
    route_payload: dict[str, Any]


class HeuristicCaptureProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    def transcribe(self, payload: bytes, filename: str | None) -> tuple[str, str]:
        try:
            decoded = payload.decode("utf-8").strip()
            if decoded:
                return decoded, "stub-text"
        except UnicodeDecodeError:
            pass

        basename = (filename or "voice note").rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
        transcript = f"Audio capture from {basename or 'recording'}."
        return transcript, "stub-generated"

    def classify(self, transcript: str, mode: CaptureMode) -> ProviderResult:
        lowered = transcript.lower().strip()

        if mode == CaptureMode.search:
            query = normalize_search_query(transcript)
            return ProviderResult(
                transcript=transcript,
                provider="heuristic",
                route=RoutingTarget.search,
                route_confidence=0.99,
                route_payload=SearchDraftPayload(query_text=query).model_dump(),
            )
        if mode == CaptureMode.note:
            payload = NoteDraftPayload(
                title=extract_title(transcript, "Untitled note"),
                body_markdown=transcript.strip(),
                tags=extract_tags(transcript),
            )
            return ProviderResult(transcript, "heuristic", RoutingTarget.note, 0.99, payload.model_dump())
        if mode == CaptureMode.todo:
            payload = TodoDraftPayload(
                title=extract_title(transcript, "Untitled todo"),
                description_markdown=transcript.strip(),
                priority=TodoPriority.high if any(token in lowered for token in ("urgent", "asap")) else TodoPriority.normal,
                deadline_at=infer_deadline(transcript),
                tags=extract_tags(transcript),
            )
            return ProviderResult(transcript, "heuristic", RoutingTarget.todo, 0.99, payload.model_dump(mode="json"))

        search_signals = ("find ", "search ", "search for", "look up", "look for", "where is", "show me")
        todo_signals = ("remember to", "remind me", "need to", "todo", "file", "send", "call", "buy", "schedule")
        if lowered.startswith(search_signals) or lowered.endswith("?"):
            query = normalize_search_query(transcript)
            return ProviderResult(
                transcript,
                "heuristic",
                RoutingTarget.search,
                0.93,
                SearchDraftPayload(query_text=query).model_dump(),
            )
        if any(signal in lowered for signal in todo_signals):
            payload = TodoDraftPayload(
                title=extract_title(transcript, "Untitled todo"),
                description_markdown=transcript.strip(),
                priority=TodoPriority.high if any(token in lowered for token in ("urgent", "asap")) else TodoPriority.normal,
                deadline_at=infer_deadline(transcript),
                tags=extract_tags(transcript),
            )
            confidence = 0.9 if "remember to" in lowered or "remind me" in lowered else 0.78
            return ProviderResult(transcript, "heuristic", RoutingTarget.todo, confidence, payload.model_dump(mode="json"))

        payload = NoteDraftPayload(
            title=extract_title(transcript, "Untitled note"),
            body_markdown=transcript.strip(),
            tags=extract_tags(transcript),
        )
        confidence = 0.65 if transcript.count(" ") < 4 else 0.87
        return ProviderResult(transcript, "heuristic", RoutingTarget.note, confidence, payload.model_dump())


class OpenAICaptureProvider(HeuristicCaptureProvider):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def transcribe(self, payload: bytes, filename: str | None) -> tuple[str, str]:
        if not self.settings.openai_api_key:
            return super().transcribe(payload, filename)
        try:
            import io

            fileobj = io.BytesIO(payload)
            fileobj.name = filename or "capture.m4a"
            response = self.client.audio.transcriptions.create(
                model=self.settings.blackhole_transcription_model,
                file=fileobj,
            )
            text = getattr(response, "text", None) or str(response)
            return text.strip(), "openai"
        except Exception:
            return super().transcribe(payload, filename)

    def classify(self, transcript: str, mode: CaptureMode) -> ProviderResult:
        if not self.settings.openai_api_key or mode != CaptureMode.auto:
            return super().classify(transcript, mode)

        system_prompt = """
You classify transcripts for a capture app. Return strict JSON with keys:
route, route_confidence, route_payload.
route must be one of note, todo, search.
For note payload keys: title, body_markdown, tags.
For todo payload keys: title, description_markdown, priority, deadline_at, tags.
For search payload keys: query_text.
Use high confidence only when the intent is obvious.
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.blackhole_router_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            route = RoutingTarget(parsed["route"])
            confidence = float(parsed["route_confidence"])
            route_payload = parsed["route_payload"]
            return ProviderResult(transcript, "openai", route, confidence, route_payload)
        except Exception:
            return super().classify(transcript, mode)


def get_capture_provider() -> HeuristicCaptureProvider:
    settings = get_settings()
    if settings.openai_api_key:
        return OpenAICaptureProvider()
    return HeuristicCaptureProvider()


def fallback_note_result(transcript: str) -> ProviderResult:
    payload = NoteDraftPayload(
        title=extract_title(transcript, "Untitled note"),
        body_markdown=transcript.strip(),
        tags=extract_tags(transcript),
    )
    return ProviderResult(
        transcript=transcript,
        provider="fallback",
        route=RoutingTarget.note,
        route_confidence=0.0,
        route_payload=payload.model_dump(),
    )


def require_owned_note(db: Session, user: User, note_id: str) -> BlackholeNote:
    note = db.scalar(select(BlackholeNote).where(BlackholeNote.id == note_id, BlackholeNote.user_id == user.id))
    if note is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    return note


def require_owned_todo(db: Session, user: User, todo_id: str) -> BlackholeTodo:
    todo = db.scalar(select(BlackholeTodo).where(BlackholeTodo.id == todo_id, BlackholeTodo.user_id == user.id))
    if todo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Todo not found")
    return todo


def require_owned_capture(db: Session, user: User, capture_id: str) -> BlackholeCapture:
    capture = db.scalar(select(BlackholeCapture).where(BlackholeCapture.id == capture_id, BlackholeCapture.user_id == user.id))
    if capture is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Capture not found")
    return capture


def require_owned_session(db: Session, user: User, session_id: str) -> BlackholeTranscriptionSession:
    transcription_session = db.scalar(
        select(BlackholeTranscriptionSession).where(
            BlackholeTranscriptionSession.id == session_id,
            BlackholeTranscriptionSession.user_id == user.id,
        )
    )
    if transcription_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcription session not found")
    return transcription_session


def create_note(db: Session, user: User, payload: NoteCreate, background_tasks: BackgroundTasks | None = None) -> BlackholeNote:
    capture = None
    if payload.source_capture_id:
        capture = require_owned_capture(db, user, payload.source_capture_id)

    note = BlackholeNote(
        user_id=user.id,
        title=payload.title.strip() or "Untitled note",
        body_markdown=payload.body_markdown.strip(),
        tags=payload.tags,
        source_capture_id=payload.source_capture_id,
    )
    db.add(note)
    db.flush()

    if capture is not None:
        capture.created_record_id = note.id
        db.add(capture)
        if background_tasks:
            background_tasks.add_task(AudioStorage().delete, capture.audio_object_key)

    db.commit()
    db.refresh(note)
    note_creation_total.inc()
    return note


def update_note(db: Session, user: User, note_id: str, payload: NoteUpdate) -> BlackholeNote:
    note = require_owned_note(db, user, note_id)
    if payload.title is not None:
        note.title = payload.title.strip() or note.title
    if payload.body_markdown is not None:
        note.body_markdown = payload.body_markdown
    if payload.tags is not None:
        note.tags = payload.tags
    if payload.archived is not None:
        note.archived_at = utcnow() if payload.archived else None
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


def list_notes(db: Session, user: User, cursor: str | None, limit: int | None) -> NotePage:
    offset = decode_cursor(cursor)
    effective_limit = clamp_limit(limit)
    stmt = (
        select(BlackholeNote)
        .where(BlackholeNote.user_id == user.id, BlackholeNote.archived_at.is_(None))
        .order_by(desc(BlackholeNote.created_at), desc(BlackholeNote.id))
        .offset(offset)
        .limit(effective_limit + 1)
    )
    rows = list(db.scalars(stmt))
    next_cursor = encode_cursor(offset + effective_limit) if len(rows) > effective_limit else None
    items = rows[:effective_limit]
    return NotePage(items=[NoteRead.model_validate(item) for item in items], next_cursor=next_cursor)


def create_todo(db: Session, user: User, payload: TodoCreate, background_tasks: BackgroundTasks | None = None) -> BlackholeTodo:
    capture = None
    if payload.source_capture_id:
        capture = require_owned_capture(db, user, payload.source_capture_id)

    todo = BlackholeTodo(
        user_id=user.id,
        title=payload.title.strip() or "Untitled todo",
        description_markdown=payload.description_markdown,
        priority=payload.priority,
        deadline_at=payload.deadline_at,
        tags=payload.tags,
        source_capture_id=payload.source_capture_id,
    )
    db.add(todo)
    db.flush()

    if capture is not None:
        capture.created_record_id = todo.id
        db.add(capture)
        if background_tasks:
            background_tasks.add_task(AudioStorage().delete, capture.audio_object_key)

    db.commit()
    db.refresh(todo)
    todo_creation_total.inc()
    return todo


def update_todo(db: Session, user: User, todo_id: str, payload: TodoUpdate) -> BlackholeTodo:
    todo = require_owned_todo(db, user, todo_id)
    fields = payload.model_fields_set
    if "title" in fields and payload.title is not None:
        todo.title = payload.title.strip() or todo.title
    if "description_markdown" in fields and payload.description_markdown is not None:
        todo.description_markdown = payload.description_markdown
    if "priority" in fields and payload.priority is not None:
        todo.priority = payload.priority
    if "deadline_at" in fields:
        todo.deadline_at = payload.deadline_at
    if "tags" in fields and payload.tags is not None:
        todo.tags = payload.tags
    if "status" in fields and payload.status is not None:
        todo.status = payload.status
        todo.completed_at = utcnow() if payload.status == TodoStatus.completed else None
    db.add(todo)
    db.commit()
    db.refresh(todo)
    return todo


def list_todos(db: Session, user: User, status_filter: str | None) -> TodoPage:
    stmt = select(BlackholeTodo).where(BlackholeTodo.user_id == user.id)
    if status_filter and status_filter.lower() != "all":
        stmt = stmt.where(BlackholeTodo.status == TodoStatus(status_filter))
    stmt = stmt.order_by(
        case((BlackholeTodo.status == TodoStatus.open, 0), else_=1),
        desc(BlackholeTodo.updated_at),
    )
    items = list(db.scalars(stmt))
    return TodoPage(items=[TodoRead.model_validate(item) for item in items], next_cursor=None)


def make_snippet(body: str, query: str) -> str:
    if not body:
        return ""
    lowered = body.lower()
    index = lowered.find(query.lower())
    if index < 0:
        return body[:160]
    start = max(0, index - 40)
    end = min(len(body), index + 120)
    return body[start:end].strip()


def search_notes(db: Session, user: User, query: str, cursor: str | None, limit: int | None) -> SearchPage:
    started = time.perf_counter()
    normalized_query = query.strip()
    if not normalized_query:
        return SearchPage(items=[], next_cursor=None)

    effective_limit = clamp_limit(limit)
    offset = decode_cursor(cursor)
    dialect = db.bind.dialect.name if db.bind is not None else "sqlite"

    if dialect == "postgresql":
        tags_text = func.coalesce(func.cast(BlackholeNote.tags, String), "")
        title_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(BlackholeNote.title, "")),
            literal_column("'A'"),
        )
        tags_vector = func.setweight(
            func.to_tsvector("english", tags_text),
            literal_column("'B'"),
        )
        body_vector = func.setweight(
            func.to_tsvector("english", func.coalesce(BlackholeNote.body_markdown, "")),
            literal_column("'C'"),
        )
        vector = title_vector.op("||")(tags_vector).op("||")(body_vector)
        ts_query = func.plainto_tsquery("english", normalized_query)
        stmt = (
            select(
                BlackholeNote,
                func.ts_rank(vector, ts_query).label("score"),
            )
            .where(
                BlackholeNote.user_id == user.id,
                BlackholeNote.archived_at.is_(None),
                vector.op("@@")(ts_query),
            )
            .order_by(desc("score"), desc(BlackholeNote.updated_at))
            .offset(offset)
            .limit(effective_limit + 1)
        )
        rows = db.execute(stmt).all()
        items = [
            SearchResult(
                id=note.id,
                title=note.title,
                snippet=make_snippet(note.body_markdown, normalized_query),
                tags=note.tags,
                score=float(score or 0.0),
                updated_at=note.updated_at,
            )
            for note, score in rows[:effective_limit]
        ]
        next_cursor = encode_cursor(offset + effective_limit) if len(rows) > effective_limit else None
        page = SearchPage(items=items, next_cursor=next_cursor)
        search_latency_seconds.observe(time.perf_counter() - started)
        if not items:
            zero_result_search_total.inc()
        return page

    terms = [term for term in re.split(r"\s+", normalized_query.lower()) if term]
    title_field = func.lower(BlackholeNote.title)
    body_field = func.lower(BlackholeNote.body_markdown)
    tags_field = func.lower(func.coalesce(func.json_extract(BlackholeNote.tags, "$"), ""))
    filters = []
    for term in terms:
        like_term = f"%{term}%"
        filters.append(or_(title_field.like(like_term), body_field.like(like_term), tags_field.like(like_term)))

    stmt = (
        select(BlackholeNote)
        .where(BlackholeNote.user_id == user.id, BlackholeNote.archived_at.is_(None))
        .where(or_(*filters))
    )
    candidates = list(db.scalars(stmt))
    ranked: list[SearchResult] = []
    for note in candidates:
        score = 0.0
        haystack_title = note.title.lower()
        haystack_body = note.body_markdown.lower()
        haystack_tags = " ".join(note.tags).lower()
        for term in terms:
            if term in haystack_title:
                score += 3
            if term in haystack_tags:
                score += 2
            if term in haystack_body:
                score += 1
        ranked.append(
            SearchResult(
                id=note.id,
                title=note.title,
                snippet=make_snippet(note.body_markdown, normalized_query),
                tags=note.tags,
                score=score,
                updated_at=note.updated_at,
            )
        )

    ranked.sort(key=lambda item: (item.score, item.updated_at), reverse=True)
    sliced = ranked[offset : offset + effective_limit + 1]
    next_cursor = encode_cursor(offset + effective_limit) if len(sliced) > effective_limit else None
    page = SearchPage(items=sliced[:effective_limit], next_cursor=next_cursor)
    search_latency_seconds.observe(time.perf_counter() - started)
    if not page.items:
        zero_result_search_total.inc()
    return page


def create_transcription_session(db: Session, user: User, mode: CaptureMode) -> BlackholeTranscriptionSession:
    transcription_session = BlackholeTranscriptionSession(user_id=user.id, mode=mode, status=TranscriptionSessionStatus.created)
    db.add(transcription_session)
    db.commit()
    db.refresh(transcription_session)
    return transcription_session


def process_transcription_session(session_id: str, audio_bytes: bytes, filename: str | None) -> None:
    provider = get_capture_provider()
    storage = AudioStorage()
    settings = get_settings()
    started = time.perf_counter()

    with SessionLocal() as db:
        transcription_session = db.scalar(
            select(BlackholeTranscriptionSession).where(BlackholeTranscriptionSession.id == session_id)
        )
        if transcription_session is None:
            return

        try:
            logger.info("transcription_session_started", extra={"session_id": session_id})
            transcription_session.status = TranscriptionSessionStatus.uploading
            db.add(transcription_session)
            db.commit()

            object_key = storage.save_bytes(user_id=transcription_session.user_id, filename=filename, payload=audio_bytes)
            transcription_session.audio_object_key = object_key
            db.add(transcription_session)
            db.commit()

            transcription_session.status = TranscriptionSessionStatus.transcribing
            db.add(transcription_session)
            db.commit()

            transcript, transcript_provider = provider.transcribe(audio_bytes, filename)
            try:
                provider_result = provider.classify(transcript, transcription_session.mode)
            except Exception:
                provider_result = fallback_note_result(transcript)

            capture_id = None
            if provider_result.route != RoutingTarget.search:
                capture = BlackholeCapture(
                    user_id=transcription_session.user_id,
                    mode=CaptureRecordMode(provider_result.route.value),
                    audio_object_key=object_key,
                    transcript_text=provider_result.transcript,
                    transcript_provider=transcript_provider,
                    routing_target=provider_result.route,
                    routing_confidence=provider_result.route_confidence,
                    routing_payload=provider_result.route_payload,
                )
                db.add(capture)
                db.commit()
                db.refresh(capture)
                capture_id = capture.id
            else:
                storage.delete(object_key)
                transcription_session.audio_object_key = None

            transcription_session.status = TranscriptionSessionStatus.completed
            transcription_session.final_transcript = provider_result.transcript
            transcription_session.route = provider_result.route
            transcription_session.route_confidence = provider_result.route_confidence
            transcription_session.route_payload = provider_result.route_payload
            transcription_session.capture_id = capture_id
            db.add(transcription_session)
            db.commit()
            transcription_latency_seconds.observe(time.perf_counter() - started)
            transcription_sessions_total.labels(status="completed").inc()
            route_distribution_total.labels(route=provider_result.route.value).inc()
            if provider_result.route_confidence < settings.blackhole_auto_route_threshold:
                low_confidence_total.inc()
            log_audit(
                db,
                "transcription.completed",
                f"session={transcription_session.id} route={provider_result.route.value} confidence={provider_result.route_confidence:.2f}",
            )
            logger.info(
                "transcription_session_completed",
                extra={
                    "session_id": transcription_session.id,
                    "route": provider_result.route.value,
                    "route_confidence": provider_result.route_confidence,
                },
            )
        except Exception as exc:
            transcription_session.status = TranscriptionSessionStatus.failed
            transcription_session.error_message = str(exc)
            db.add(transcription_session)
            db.commit()
            transcription_latency_seconds.observe(time.perf_counter() - started)
            transcription_sessions_total.labels(status="failed").inc()
            log_audit(db, "transcription.failed", f"session={transcription_session.id} error={exc}")
            logger.exception("transcription_session_failed", extra={"session_id": transcription_session.id})


def build_final_result(transcription_session: BlackholeTranscriptionSession) -> TranscriptionFinalResult:
    return TranscriptionFinalResult(
        session_id=transcription_session.id,
        transcript=transcription_session.final_transcript or "",
        route=transcription_session.route or RoutingTarget.note,
        route_confidence=transcription_session.route_confidence or 0.0,
        route_payload=transcription_session.route_payload or {},
        capture_id=transcription_session.capture_id,
    )


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_transcription_events(session_id: str, user_id: str) -> Any:
    sent: set[str] = set()

    yield sse_message("session_created", {"session_id": session_id})

    while True:
        with SessionLocal() as db:
            transcription_session = db.scalar(
                select(BlackholeTranscriptionSession).where(
                    BlackholeTranscriptionSession.id == session_id,
                    BlackholeTranscriptionSession.user_id == user_id,
                )
            )
            if transcription_session is None:
                yield sse_message("error", {"message": "Session not found"})
                return

            if transcription_session.status in {
                TranscriptionSessionStatus.uploading,
                TranscriptionSessionStatus.transcribing,
                TranscriptionSessionStatus.completed,
                TranscriptionSessionStatus.failed,
            } and "uploading" not in sent:
                sent.add("uploading")
                yield sse_message("uploading", {"session_id": session_id})

            if transcription_session.status in {
                TranscriptionSessionStatus.transcribing,
                TranscriptionSessionStatus.completed,
                TranscriptionSessionStatus.failed,
            } and "transcribing" not in sent:
                sent.add("transcribing")
                yield sse_message("transcribing", {"session_id": session_id})

            if transcription_session.status == TranscriptionSessionStatus.completed:
                yield sse_message("final_result", build_final_result(transcription_session).model_dump(mode="json"))
                return

            if transcription_session.status == TranscriptionSessionStatus.failed:
                yield sse_message("error", {"message": transcription_session.error_message or "Transcription failed"})
                return

        await asyncio.sleep(0.35)


def cleanup_expired_audio(db: Session) -> int:
    settings = get_settings()
    storage = AudioStorage()
    cutoff = utcnow() - timedelta(hours=settings.blackhole_audio_retention_hours)
    deleted = 0
    deleted_keys: set[str] = set()

    captures = list(db.scalars(select(BlackholeCapture).where(BlackholeCapture.created_at < cutoff)))
    for capture in captures:
        if capture.audio_object_key not in deleted_keys:
            storage.delete(capture.audio_object_key)
            deleted_keys.add(capture.audio_object_key)
            deleted += 1

    sessions = list(
        db.scalars(
            select(BlackholeTranscriptionSession).where(
                BlackholeTranscriptionSession.created_at < cutoff,
                BlackholeTranscriptionSession.status.in_(
                    [TranscriptionSessionStatus.completed, TranscriptionSessionStatus.failed]
                ),
            )
        )
    )
    for transcription_session in sessions:
        if (
            transcription_session.audio_object_key
            and transcription_session.audio_object_key not in deleted_keys
            and (
                transcription_session.route == RoutingTarget.search
                or transcription_session.status == TranscriptionSessionStatus.failed
            )
        ):
            storage.delete(transcription_session.audio_object_key)
            deleted_keys.add(transcription_session.audio_object_key)
            deleted += 1

    db.commit()
    return deleted
