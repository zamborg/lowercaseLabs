from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ItemType = Literal[
    "note",
    "todo",
    "event",
    "epic",
    "contact",
    "resource",
    "decision",
    "journal",
    "habit",
    "table",
]
PublicItemType = Literal[
    "note",
    "todo",
    "event",
    "epic",
    "contact",
    "resource",
    "decision",
    "journal",
    "habit",
]
ItemStatus = Literal["open", "in_progress", "done", "cancelled", "archived"]
ItemPriority = Literal["low", "medium", "high", "urgent"]
ItemSource = Literal["voice", "manual", "agent", "import", "system"]
ReadStatus = Literal["unread", "reading", "read", "skipped"]
LinkType = Literal["reference", "blocks", "spawned_from", "relates_to", "contradicts"]
TableColumnType = Literal["text", "number", "boolean", "date", "select"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppleSignInRequest(StrictBaseModel):
    identity_token: str


class CaptureRequest(StrictBaseModel):
    content: str = Field(min_length=1)
    source: ItemSource = "voice"


class ItemCreateRequest(StrictBaseModel):
    type: Optional[PublicItemType] = None
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[ItemStatus] = None
    priority: Optional[ItemPriority] = None
    source: Optional[ItemSource] = None
    parent_id: Optional[str] = None
    epic_id: Optional[str] = None
    completed: Optional[bool] = None
    tags: Optional[list[str]] = None
    due_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    read_status: Optional[ReadStatus] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    organization: Optional[str] = None
    recurrence_rule: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class UpdateItemRequest(StrictBaseModel):
    completed: Optional[bool] = None
    title: Optional[str] = None
    content: Optional[str] = None
    type: Optional[PublicItemType] = None
    status: Optional[ItemStatus] = None
    priority: Optional[ItemPriority] = None
    source: Optional[ItemSource] = None
    parent_id: Optional[str] = None
    epic_id: Optional[str] = None
    due_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    read_status: Optional[ReadStatus] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    organization: Optional[str] = None
    recurrence_rule: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None


class Item(BaseModel):
    id: str
    type: ItemType
    title: str
    content: str
    status: ItemStatus
    priority: Optional[ItemPriority] = None
    source: ItemSource
    parent_id: Optional[str] = None
    epic_id: Optional[str] = None
    completed: bool
    tags: list[str]
    due_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    read_status: Optional[ReadStatus] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    organization: Optional[str] = None
    recurrence_rule: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ItemListResponse(BaseModel):
    items: list[Item]
    limit: int
    offset: int
    has_more: bool


class CaptureResponse(BaseModel):
    items_created: list[Item]
    llm_log_id: Optional[str] = None


class SearchRequest(StrictBaseModel):
    query: str


class TableColumn(StrictBaseModel):
    name: str
    type: TableColumnType = "text"
    options: Optional[list[str]] = None


class TableCreateRequest(StrictBaseModel):
    title: str
    description: Optional[str] = None
    columns: list[TableColumn]
    parent_id: Optional[str] = None
    tags: Optional[list[str]] = None


class TableUpdateRequest(StrictBaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class Table(BaseModel):
    id: str
    item_id: str
    title: str
    description: Optional[str] = None
    columns: list[dict[str, Any]]
    created_at: str
    updated_at: str


class TableRowCreateRequest(StrictBaseModel):
    data: dict[str, Any]


class TableRowUpdateRequest(StrictBaseModel):
    data: dict[str, Any]


class TableRow(BaseModel):
    id: str
    table_id: str
    data: dict[str, Any]
    row_order: int
    created_at: str
    updated_at: str


class ListEnvelope(BaseModel):
    items: list[Any]
    limit: int
    offset: int
    has_more: bool


class TableDetailResponse(BaseModel):
    table: Table
    rows: list[TableRow]


class LinkCreateRequest(StrictBaseModel):
    source_id: str
    target_id: str
    link_type: LinkType


class ItemLink(BaseModel):
    id: str
    source_id: str
    target_id: str
    link_type: LinkType
    created_at: str


class HabitCompletionCreateRequest(StrictBaseModel):
    completed_on: Optional[str] = None
    completed_at: Optional[str] = None
    note: Optional[str] = None


class HabitCompletion(BaseModel):
    id: str
    item_id: str
    completed_on: str
    completed_at: str
    note: Optional[str] = None
    created_at: str


class ApiKeyCreateRequest(StrictBaseModel):
    label: Optional[str] = None


class ApiKeyCreateResponse(BaseModel):
    id: str
    key: str
    label: Optional[str] = None
    created_at: str
    last_used_at: Optional[str] = None


class ApiKey(BaseModel):
    id: str
    label: Optional[str] = None
    created_at: str
    last_used_at: Optional[str] = None


class AgentRequest(StrictBaseModel):
    message: str


class AgentToolCallSummary(BaseModel):
    name: str
    status: str


class AgentResponse(BaseModel):
    run_id: str
    response: str
    items_created: list[Item] = Field(default_factory=list)
    items_updated: list[Item] = Field(default_factory=list)
    tool_calls: list[AgentToolCallSummary] = Field(default_factory=list)


class AgentBriefResponse(BaseModel):
    response: str


class AgentLintFinding(BaseModel):
    type: str
    item_id: Optional[str] = None
    detail: str


class AgentLintResponse(BaseModel):
    findings: list[AgentLintFinding]
    summary: str


class MeResponse(BaseModel):
    id: str
    item_counts: dict[str, int]


class TagsResponse(BaseModel):
    tags: list[str]
