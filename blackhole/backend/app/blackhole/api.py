from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.blackhole.models import CaptureMode
from app.blackhole.schemas import (
    NoteCreate,
    NotePage,
    NoteRead,
    NoteUpdate,
    SearchPage,
    TodoCreate,
    TodoPage,
    TodoRead,
    TodoUpdate,
    TranscriptionSessionCreateResponse,
    TranscriptionSessionRead,
)
from app.blackhole.services import (
    cleanup_expired_audio,
    create_note,
    create_todo,
    create_transcription_session,
    list_notes,
    list_todos,
    process_transcription_session,
    require_owned_note,
    require_owned_session,
    require_owned_todo,
    search_notes,
    stream_transcription_events,
    update_note,
    update_todo,
)
from app.db import get_db
from app.models import User
from app.schemas import APIMessage

router = APIRouter(prefix="/blackhole", tags=["blackhole"])


@router.post("/transcriptions", response_model=TranscriptionSessionCreateResponse)
async def post_transcription(
    background_tasks: BackgroundTasks,
    mode: str = Form("auto"),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptionSessionCreateResponse:
    try:
        transcription_mode = CaptureMode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid mode") from exc
    transcription_session = create_transcription_session(db, current_user, transcription_mode)
    payload = await audio.read()
    background_tasks.add_task(process_transcription_session, transcription_session.id, payload, audio.filename)
    return TranscriptionSessionCreateResponse(transcription_session_id=transcription_session.id)


@router.get("/transcriptions/{session_id}", response_model=TranscriptionSessionRead)
def get_transcription_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TranscriptionSessionRead:
    return TranscriptionSessionRead.model_validate(require_owned_session(db, current_user, session_id))


@router.get("/transcriptions/{session_id}/events")
async def get_transcription_events(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    require_owned_session(db, current_user, session_id)
    return StreamingResponse(stream_transcription_events(session_id, current_user.id), media_type="text/event-stream")


@router.post("/notes", response_model=NoteRead)
def post_note(
    payload: NoteCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NoteRead:
    return NoteRead.model_validate(create_note(db, current_user, payload, background_tasks))


@router.get("/notes", response_model=NotePage)
def get_notes(
    cursor: str | None = None,
    limit: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NotePage:
    return list_notes(db, current_user, cursor, limit)


@router.get("/notes/{note_id}", response_model=NoteRead)
def get_note(
    note_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NoteRead:
    return NoteRead.model_validate(require_owned_note(db, current_user, note_id))


@router.patch("/notes/{note_id}", response_model=NoteRead)
def patch_note(
    note_id: str,
    payload: NoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> NoteRead:
    return NoteRead.model_validate(update_note(db, current_user, note_id, payload))


@router.post("/todos", response_model=TodoRead)
def post_todo(
    payload: TodoCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TodoRead:
    return TodoRead.model_validate(create_todo(db, current_user, payload, background_tasks))


@router.get("/todos", response_model=TodoPage)
def get_todos(
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TodoPage:
    return list_todos(db, current_user, status)


@router.get("/todos/{todo_id}", response_model=TodoRead)
def get_todo(
    todo_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TodoRead:
    return TodoRead.model_validate(require_owned_todo(db, current_user, todo_id))


@router.patch("/todos/{todo_id}", response_model=TodoRead)
def patch_todo(
    todo_id: str,
    payload: TodoUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TodoRead:
    return TodoRead.model_validate(update_todo(db, current_user, todo_id, payload))


@router.get("/search", response_model=SearchPage)
def get_search(
    q: str,
    limit: int | None = None,
    cursor: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SearchPage:
    return search_notes(db, current_user, q, cursor, limit)


@router.post("/admin/run_audio_cleanup", response_model=APIMessage)
def run_audio_cleanup(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> APIMessage:
    deleted = cleanup_expired_audio(db)
    return APIMessage(message=f"deleted={deleted}")
