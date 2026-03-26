from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.blackhole.models import CaptureMode, RoutingTarget, TodoPriority, TodoStatus


class CursorPage(BaseModel):
    next_cursor: str | None = None


class NoteDraftPayload(BaseModel):
    title: str
    body_markdown: str
    tags: list[str] = Field(default_factory=list)


class TodoDraftPayload(BaseModel):
    title: str
    description_markdown: str = ""
    priority: TodoPriority = TodoPriority.normal
    deadline_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class SearchDraftPayload(BaseModel):
    query_text: str


class NoteCreate(BaseModel):
    title: str
    body_markdown: str
    tags: list[str] = Field(default_factory=list)
    source_capture_id: str | None = None


class NoteUpdate(BaseModel):
    title: str | None = None
    body_markdown: str | None = None
    tags: list[str] | None = None
    archived: bool | None = None


class NoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str
    body_markdown: str
    tags: list[str]
    source_capture_id: str | None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None


class NotePage(CursorPage):
    items: list[NoteRead]


class TodoCreate(BaseModel):
    title: str
    description_markdown: str = ""
    priority: TodoPriority = TodoPriority.normal
    deadline_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    source_capture_id: str | None = None


class TodoUpdate(BaseModel):
    title: str | None = None
    description_markdown: str | None = None
    priority: TodoPriority | None = None
    deadline_at: datetime | None = None
    tags: list[str] | None = None
    status: TodoStatus | None = None


class TodoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str
    description_markdown: str
    priority: TodoPriority
    status: TodoStatus
    deadline_at: datetime | None
    tags: list[str]
    source_capture_id: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class TodoPage(CursorPage):
    items: list[TodoRead]


class SearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    tags: list[str]
    score: float
    updated_at: datetime


class SearchPage(CursorPage):
    items: list[SearchResult]


class TranscriptionSessionCreateResponse(BaseModel):
    transcription_session_id: str


class TranscriptionFinalResult(BaseModel):
    session_id: str
    transcript: str
    route: RoutingTarget
    route_confidence: float
    route_payload: dict[str, object]
    capture_id: str | None = None


class TranscriptionSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    mode: CaptureMode
    status: str
    audio_object_key: str | None
    final_transcript: str | None
    route: RoutingTarget | None
    route_confidence: float | None
    route_payload: dict[str, object] | None
    capture_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

