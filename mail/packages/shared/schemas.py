from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmailAddress(BaseModel):
    name: Optional[str] = None
    email: str


class CategorySuggestion(BaseModel):
    id: str
    name: str
    confidence: float


class AgentHint(BaseModel):
    job_id: str
    type: str
    status: str
    summary: Optional[str] = None


class EmailSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    subject: Optional[str] = None
    snippet: Optional[str] = None
    date: Optional[str] = None
    from_: List[EmailAddress] = Field(default_factory=list, alias="from")
    to: List[EmailAddress] = Field(default_factory=list)
    is_read: bool
    loop_status: str
    category_suggestion: Optional[CategorySuggestion] = None
    agent_hint: Optional[AgentHint] = None


class ThreadMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    subject: Optional[str] = None
    date: Optional[str] = None
    from_: List[EmailAddress] = Field(default_factory=list, alias="from")
    snippet: Optional[str] = None


class EmailDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    subject: Optional[str] = None
    date: Optional[str] = None
    from_: List[EmailAddress] = Field(default_factory=list, alias="from")
    to: List[EmailAddress] = Field(default_factory=list)
    cc: List[EmailAddress] = Field(default_factory=list)
    bcc: List[EmailAddress] = Field(default_factory=list)
    snippet: Optional[str] = None
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    labels: Dict[str, Any] = Field(default_factory=dict)
    is_read: bool
    thread_context: List[ThreadMessage] = Field(default_factory=list)


class CategoryIn(BaseModel):
    name: str
    description: Optional[str] = None
    matching_rules: Dict[str, Any] = Field(default_factory=dict)
    default_policy_id: Optional[str] = None


class CategoryOut(CategoryIn):
    id: str
    user_id: str


class PolicyIn(BaseModel):
    name: str
    autonomy_level: int = Field(ge=0, le=3)
    allowed_actions: List[str] = Field(default_factory=list)
    reply_template: Optional[str] = None
    style_profile: Dict[str, Any] = Field(default_factory=dict)
    escalation_rules: Dict[str, Any] = Field(default_factory=dict)


class PolicyOut(PolicyIn):
    id: str
    user_id: str


class AgentJob(BaseModel):
    id: str
    user_id: str
    type: str
    status: str
    input_refs: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DraftResponse(BaseModel):
    job_id: str
    status: str
    draft_text: Optional[str] = None
    rationale_short: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    policy_checks: List[str] = Field(default_factory=list)


class SnoozeRequest(BaseModel):
    until: str
    reason: Optional[str] = None


class RememberRequest(BaseModel):
    note_category: Optional[str] = None
    why: Optional[str] = None


class DelegateRequest(BaseModel):
    instruction: str
    category_id: Optional[str] = None
    autonomy_override: Optional[int] = Field(default=None, ge=0, le=3)


class NotesDocument(BaseModel):
    id: str
    user_id: str
    title: str
    content_markdown: str
    updated_at: Optional[str] = None


class NotesItem(BaseModel):
    id: str
    notes_document_id: str
    email_id: str
    summary: str
    tags: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None
