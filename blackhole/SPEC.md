# blackhole - V2 spec

This document is the canonical V2 implementation contract for blackhole. Build from
this. If something here conflicts with existing code, this spec wins.

V2 is a foundation release, not the final product. The priority is a data model and
API surface that can be tested heavily, extended safely, and used by both the iOS app
and future automation. Prefer boring, explicit contracts over clever behavior.

## 0. V2 scope

### Goals

1. Expand blackhole from note/todo/epic capture into a structured personal data
   backend.
2. Keep fast voice capture low-latency.
3. Add deterministic HTTP APIs for items, tables, links, habits, API keys, and a
   single-turn agent.
4. Make every write path enforce the same domain invariants.
5. Preserve current iOS/backend data during migration.

### Non-goals for V2

These are intentionally out of scope:

1. Streaming agent responses.
2. Persistent multi-turn chat threads.
3. Anthropic migration. V2 reuses the existing OpenAI Responses path.
4. Production `run_python`.
5. Proactive auto-linking.
6. Advanced recurrence expansion.
7. Table formulas, joins, column migrations, or computed columns.
8. MCP server implementation before HTTP contracts are stable.

## 1. Architecture contract

There are two ingestion paths. Do not merge them.

### 1.1 Capture path

`POST /captures` is the fast LLM classification path.

```
iOS voice/text capture
  -> POST /captures {content}
  -> OpenAI fast structured-output classifier
  -> domain service validates returned item drafts
  -> INSERT item(s)
  -> return CaptureResponse
```

Characteristics:

1. Optimized for latency.
2. No tool loop.
3. No agent reasoning.
4. Creates the minimum useful set of durable items.
5. Uses existing OpenAI Responses infrastructure.

Legacy compatibility:

1. Current iOS calls `POST /items` with only `{content}`.
2. During the transition, `POST /items` MAY continue to accept the legacy capture
   shape when `type` is omitted.
3. New clients MUST use `POST /captures` for LLM classification.
4. `POST /items` is otherwise deterministic CRUD.

### 1.2 Deterministic CRUD path

`POST /items`, `PATCH /items/{item_id}`, table endpoints, link endpoints, and habit
completion endpoints are deterministic. They never call an LLM.

### 1.3 Agent path

`POST /agent` is the single-turn agent path.

```
user message
  -> POST /agent {message}
  -> build and persist context snapshot
  -> OpenAI agent model with tool loop
  -> domain service handles all writes
  -> persist agent run + tool calls
  -> return AgentResponse
```

Characteristics:

1. Not a chat transcript in V2. Each request is independent.
2. The system prompt is assembled fresh per request, then snapshotted into
   `agent_runs.context_json`.
3. The loop has a hard maximum of 8 tool turns.
4. Tool calls are logged individually.
5. Tool errors are returned to the model as tool results and logged.
6. If the loop exceeds max turns, return HTTP 502 with error code
   `agent_max_turns_exceeded`.

### 1.4 Domain service is the source of truth

All write paths call the same domain service functions. HTTP handlers and agent tool
handlers must not duplicate validation.

Minimum service boundary:

```
HTTP endpoint
  -> auth
  -> request schema validation
  -> domain service
  -> db.py
  -> response schema

Agent tool
  -> tool schema validation
  -> domain service
  -> db.py
  -> tool result
```

The service layer enforces invariants listed in section 3.

## 2. Data primitives

Everything is an item, a table, or a link.

### 2.1 Items

There are nine user-visible item types plus one internal backing type.

| Type | Visible | Purpose | Required fields |
|---|---:|---|---|
| `note` | yes | Durable knowledge artifact | `title`, `content` |
| `todo` | yes | Single-occurrence actionable item | `title`, `content` |
| `event` | yes | Scheduled time slot | `title`, `content`, `start_time`, `end_time` |
| `epic` | yes | Project/workstream/goal container | `title`, `content` |
| `contact` | yes | Person with relationship context | `title`, `content` |
| `resource` | yes | URL/reference/bookmark to consume | `title`, `content` |
| `decision` | yes | Recorded decision with rationale | `title`, `content` |
| `journal` | yes | Time-anchored daily prose entry | `content`; title generated if absent |
| `habit` | yes | Recurring commitment with manual completion log | `title`, `content`, `recurrence_rule` |
| `table` | internal | Backing item for a user table | created only by table service |

Classification signals:

1. `journal`: stream-of-consciousness, reflective, no clear subject or action item.
2. `note`: about a specific topic; meant to be referenced later.
3. `todo`: a single concrete action.
4. `habit`: explicitly recurring cadence such as daily, every morning, or 3x/week.
5. `decision`: a choice already made, with or without rationale.
6. `event`: a scheduled time slot with start and end.
7. `epic`: project/workstream/goal container language.
8. `contact`: person plus relationship context.
9. `resource`: URL, book, article, podcast, video, or other consumable reference.

### 2.2 Tables

Tables are user-defined structured data. A table has a backing item with
`type='table'`, so it can be tagged, searched, parented, and linked.

Table column types:

```
text | number | date | boolean | select
```

V2 table limits:

1. No formulas.
2. No joins.
3. No column rename/drop migrations.
4. Row validation is strict against the table schema.

### 2.3 Links

Links are typed directed edges between items. A link always belongs to a user.

Allowed link types:

| Link type | Meaning |
|---|---|
| `reference` | Source references or cites target |
| `blocks` | Source is blocked by target |
| `spawned_from` | Source was created because of target |
| `relates_to` | General semantic relationship |
| `contradicts` | Source conflicts with target |

There is no stored `is_blocked_by` link type. That inverse is derived by querying
links where the item is the target of a `blocks` link.

## 3. Invariants

The service layer must enforce these rules for every write path.

### 3.1 Universal item invariants

1. `type` must be one of:
   `note`, `todo`, `event`, `epic`, `contact`, `resource`, `decision`, `journal`,
   `habit`, `table`.
2. `status` must be one of:
   `open`, `in_progress`, `done`, `cancelled`, `archived`.
3. `priority` must be null or one of:
   `low`, `medium`, `high`, `urgent`.
4. `source` must be one of:
   `voice`, `manual`, `agent`, `import`, `system`.
5. `tags` must be a JSON array of non-empty strings.
6. `metadata` must be null or a JSON object.
7. All timestamps stored by the backend are UTC ISO 8601 strings.
8. Unknown write fields are rejected.
9. Omitted PATCH fields are unchanged.
10. Explicit `null` clears nullable fields.

Default values on create:

1. `status` defaults to `open` for every type.
2. `priority` defaults to null.
3. `source` defaults to `manual` for deterministic item creates and `voice` for
   captures.
4. `tags` defaults to `[]`.
5. `metadata` defaults to `{}`.
6. `resource.read_status` defaults to `unread`.

### 3.2 Status and completed compatibility

`status` is canonical.

`completed` is legacy compatibility only:

1. On migration, existing rows map to:
   `completed=1 -> status='done'`; `completed=0 -> status='open'`.
2. On every write, `completed` is kept in sync:
   `completed = 1` iff `status='done'`.
3. Legacy `PATCH /items/{item_id} {"completed": true}` maps to
   `status='done'`.
4. Legacy `PATCH /items/{item_id} {"completed": false}` maps to
   `status='open'`.
5. New code must read and write `status`, not `completed`.

### 3.3 Parent and epic compatibility

`parent_id` is canonical.

`epic_id` is legacy compatibility only:

1. On migration, if an existing row has `epic_id`, copy it into `parent_id`.
2. If `parent_id` points to an item whose `type='epic'`, responses include
   `epic_id=parent_id` for legacy clients.
3. If `parent_id` is null or points to a non-epic item, responses include
   `epic_id=null`.
4. New code must read and write `parent_id`, not `epic_id`.

Parent rules:

1. An item cannot be its own parent.
2. Parent cycles are rejected.
3. `epic` items cannot have a parent in V2.
4. `table` backing items can have a parent, but table creation owns the initial
   backing item.

### 3.4 Type-specific invariants

1. `event` requires `start_time` and `end_time`; `end_time` must be after
   `start_time`.
2. A deadline without a time slot is a `todo`, not an `event`.
3. `habit` requires `recurrence_rule`.
4. `journal` may omit title on create; the backend generates
   `Journal - YYYY-MM-DD HH:mm`.
5. `decision` content and title are immutable after creation. Only `tags`,
   `parent_id`, `status`, `priority`, and `metadata` may be patched.
6. `resource.read_status` must be null or one of:
   `unread`, `reading`, `read`, `skipped`.
7. `table` items are created by the table service only. `POST /items` with
   `type='table'` is rejected.
8. `contact` may use `email`, `phone`, `organization`, and `location`; each is
   nullable.

### 3.5 Link invariants

1. `source_id` and `target_id` must both belong to the authenticated user.
2. `source_id` cannot equal `target_id`.
3. `(source_id, target_id, link_type)` is unique.
4. Deleting an item cascades its links.

### 3.6 Habit completion invariants

1. Completion endpoints only accept items whose `type='habit'`.
2. Each completion belongs to the authenticated user.
3. `completed_on` is a local date string, `YYYY-MM-DD`.
4. There is at most one completion per `(user_id, item_id, completed_on)`.
5. If `completed_at` is omitted, backend uses current UTC time.
6. If `completed_on` is omitted, backend derives it from `completed_at` using UTC.

### 3.7 Table invariants

1. Column names are unique within a table.
2. Column names are non-empty strings.
3. Column type must be one of the allowed table column types.
4. `select` columns require a non-empty `options` array.
5. Row data keys must match known columns.
6. Missing row values are allowed and treated as null.
7. `number` values must be numeric.
8. `boolean` values must be booleans.
9. `date` values must be ISO 8601 strings.
10. `select` values must be one of the column options.

## 4. Database schema

SQLite remains the V2 datastore.

### 4.1 `users`

Existing table, unchanged.

```sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
```

### 4.2 `items`

```sql
CREATE TABLE IF NOT EXISTS items (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),

    type            TEXT NOT NULL DEFAULT 'note',
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,

    status          TEXT NOT NULL DEFAULT 'open',
    priority        TEXT,
    source          TEXT NOT NULL DEFAULT 'voice',

    parent_id       TEXT REFERENCES items(id),

    tags            TEXT NOT NULL DEFAULT '[]',
    due_date        TEXT,

    start_time      TEXT,
    end_time        TEXT,
    location        TEXT,
    url             TEXT,
    read_status     TEXT,
    email           TEXT,
    phone           TEXT,
    organization    TEXT,
    recurrence_rule TEXT,
    metadata        TEXT,

    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,

    -- legacy compatibility
    epic_id         TEXT,
    completed       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_items_user_id ON items(user_id);
CREATE INDEX IF NOT EXISTS idx_items_type_status ON items(user_id, type, status);
CREATE INDEX IF NOT EXISTS idx_items_parent_id ON items(user_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_items_start_time ON items(user_id, start_time);
CREATE INDEX IF NOT EXISTS idx_items_due_date ON items(user_id, due_date);
CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_epic_id ON items(epic_id);
```

SQLite CHECK constraints are optional in V2. The service layer is the required
enforcement point because migrations must be additive and safe.

### 4.3 `tables`

```sql
CREATE TABLE IF NOT EXISTS tables (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    item_id     TEXT NOT NULL UNIQUE REFERENCES items(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    description TEXT,
    columns     TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tables_user_id ON tables(user_id);
```

`columns` JSON shape:

```json
[
  {"name": "amount", "type": "number"},
  {"name": "category", "type": "select", "options": ["food", "travel"]}
]
```

### 4.4 `table_rows`

```sql
CREATE TABLE IF NOT EXISTS table_rows (
    id         TEXT PRIMARY KEY,
    table_id   TEXT NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
    data       TEXT NOT NULL DEFAULT '{}',
    row_order  INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_table_rows_table_id
    ON table_rows(table_id, row_order);
```

### 4.5 `item_links`

```sql
CREATE TABLE IF NOT EXISTS item_links (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    source_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    link_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE(source_id, target_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_item_links_user_source
    ON item_links(user_id, source_id);
CREATE INDEX IF NOT EXISTS idx_item_links_user_target
    ON item_links(user_id, target_id);
```

### 4.6 `habit_completions`

```sql
CREATE TABLE IF NOT EXISTS habit_completions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id),
    item_id      TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    completed_on TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    note         TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE(user_id, item_id, completed_on)
);

CREATE INDEX IF NOT EXISTS idx_habit_completions_item
    ON habit_completions(user_id, item_id, completed_on DESC);
```

### 4.7 `api_keys`

```sql
CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id),
    key_hash     TEXT NOT NULL UNIQUE,
    label        TEXT,
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
```

Raw API keys are displayed once and never stored. Raw keys use this shape:

```
bh_live_<random_urlsafe_secret>
```

### 4.8 `agent_runs`

```sql
CREATE TABLE IF NOT EXISTS agent_runs (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    message         TEXT NOT NULL,
    response        TEXT,
    context_json    TEXT NOT NULL,
    model           TEXT NOT NULL,
    status          TEXT NOT NULL,
    error           TEXT,
    tool_turns      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_user_created
    ON agent_runs(user_id, created_at DESC);
```

`status` values:

```
running | success | error | max_turns_exceeded
```

### 4.9 `agent_tool_calls`

```sql
CREATE TABLE IF NOT EXISTS agent_tool_calls (
    id             TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    user_id        TEXT NOT NULL REFERENCES users(id),
    tool_call_id   TEXT,
    name           TEXT NOT NULL,
    input_json     TEXT NOT NULL,
    output_json    TEXT,
    status         TEXT NOT NULL,
    error          TEXT,
    duration_ms    INTEGER,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run
    ON agent_tool_calls(run_id, created_at);
```

`status` values:

```
success | error
```

## 5. Migration contract

Migration must be additive and idempotent.

Required migration behavior:

1. Existing `users`, `items`, and `llm_logs` are preserved.
2. Missing `items` columns are added with `ALTER TABLE`.
3. Existing rows receive:
   `status='done'` when `completed=1`, otherwise `status='open'`.
4. Existing rows receive `source='voice'`.
5. Existing rows receive `parent_id=epic_id` when `epic_id` is not null.
6. Existing invalid tag JSON is normalized to `[]` on read; migration may leave raw
   data untouched if read paths handle it.
7. No migration deletes data.
8. Re-running `init_db()` is safe.

Compatibility warning:

Before the backend starts producing new item types in production, iOS must decode
unknown item types without crashing. Existing iOS has a closed enum for
`note|todo|epic`; that must be fixed early.

## 6. API conventions

### 6.1 Authentication

All authenticated endpoints use:

```
Authorization: Bearer <token>
```

Token dispatch:

1. Tokens beginning with `bh_` are API keys.
2. All other bearer tokens are Apple session JWTs.
3. Do not try both paths for the same token.

Session JWT:

1. Existing Apple Sign-In flow remains.
2. Session token lifetime remains 90 days unless changed elsewhere.

API key:

1. Stored as SHA-256 hash of the raw key.
2. Raw key returned only from creation endpoint.
3. `last_used_at` updates after successful API key authentication.

### 6.2 Error envelope

All non-2xx JSON errors use one envelope:

```json
{
  "error": {
    "code": "invalid_item_type",
    "message": "Item type is not supported.",
    "details": {"type": "foo"}
  }
}
```

Common status codes:

| HTTP | Meaning |
|---:|---|
| 400 | invalid request or domain invariant violation |
| 401 | missing or invalid auth |
| 403 | authenticated but not allowed |
| 404 | resource not found for this user |
| 409 | uniqueness conflict |
| 422 | schema validation failure |
| 500 | unexpected server error |
| 502 | model/tool loop failed |

### 6.3 List envelope

All V2 list endpoints return:

```json
{
  "items": [],
  "limit": 100,
  "offset": 0,
  "has_more": false
}
```

Default `limit` is 100. Maximum `limit` is 500.

### 6.4 Timestamp format

Backend-generated timestamps are UTC ISO 8601 strings with timezone information.

Clients may submit ISO 8601 timestamps with timezone information. The backend stores
them normalized to UTC where practical.

## 7. API schemas

The JSON shapes below are the V2 contract. Pydantic models should mirror these
shapes.

### 7.1 Item response

```json
{
  "id": "item-id",
  "type": "todo",
  "title": "Buy milk",
  "content": "buy milk tomorrow",
  "status": "open",
  "priority": "medium",
  "source": "voice",
  "parent_id": null,
  "epic_id": null,
  "completed": false,
  "tags": ["errand"],
  "due_date": "2026-04-29T17:00:00Z",
  "start_time": null,
  "end_time": null,
  "location": null,
  "url": null,
  "read_status": null,
  "email": null,
  "phone": null,
  "organization": null,
  "recurrence_rule": null,
  "metadata": {},
  "created_at": "2026-04-29T16:00:00Z",
  "updated_at": "2026-04-29T16:00:00Z"
}
```

### 7.2 Captures

```
POST /captures
```

Request:

```json
{
  "content": "remember to buy milk tomorrow at 5",
  "source": "voice"
}
```

`source` is optional and defaults to `voice`. Allowed source values are the universal
source enum.

Response:

```json
{
  "items_created": [Item],
  "llm_log_id": "log-id-or-null"
}
```

Classifier output contract before persistence:

```json
{
  "items": [
    {
      "type": "note|todo|event|epic|contact|resource|decision|journal|habit",
      "title": "string or null for journal",
      "content": "string or null; defaults to capture content",
      "status": "open|in_progress|done|cancelled|archived or null",
      "priority": "low|medium|high|urgent or null",
      "tags": ["string"],
      "parent_title": "string or null",
      "due_date": "ISO timestamp or null",
      "start_time": "ISO timestamp or null",
      "end_time": "ISO timestamp or null",
      "location": "string or null",
      "url": "string or null",
      "read_status": "unread|reading|read|skipped or null",
      "email": "string or null",
      "phone": "string or null",
      "organization": "string or null",
      "recurrence_rule": "object or null",
      "metadata": "object or null"
    }
  ]
}
```

The classifier may suggest `parent_title`, but the service resolves parents by exact
existing title match within the user's items. If no exact match exists, `parent_id`
is null unless the capture also created the parent item in the same response.

### 7.3 Items

```
GET /items
```

Query parameters:

| Param | Type | Description |
|---|---|---|
| `type` | string | exact item type |
| `status` | string | exact status |
| `priority` | string | exact priority |
| `parent_id` | string | children of parent |
| `tags` | string | comma-separated, item must contain all |
| `query` | string | substring match over title, content, tags |
| `limit` | int | default 100, max 500 |
| `offset` | int | default 0 |

Response:

```json
{
  "items": [Item],
  "limit": 100,
  "offset": 0,
  "has_more": false
}
```

```
GET /items/{item_id}
```

Response: `Item`.

```
POST /items
```

Deterministic create request:

```json
{
  "type": "todo",
  "title": "Buy milk",
  "content": "Buy milk tomorrow",
  "status": "open",
  "priority": "medium",
  "source": "manual",
  "parent_id": null,
  "tags": ["errand"],
  "due_date": null,
  "start_time": null,
  "end_time": null,
  "location": null,
  "url": null,
  "read_status": null,
  "email": null,
  "phone": null,
  "organization": null,
  "recurrence_rule": null,
  "metadata": {}
}
```

Response: `Item`.

Legacy create request:

```json
{"content": "raw voice capture"}
```

This is treated as `POST /captures` only during migration.

```
PATCH /items/{item_id}
```

Request: partial item fields. Omitted fields are unchanged. Null clears nullable
fields. Unknown fields are rejected.

Response: `Item`.

```
DELETE /items/{item_id}
```

Response:

```json
{"deleted": true}
```

Deleting a `table` backing item through this endpoint is rejected with
`table_item_delete_forbidden`; use `DELETE /tables/{table_id}`.

```
GET /items/{item_id}/children
GET /items/{item_id}/links
```

Children response uses the list envelope. Links response uses the list envelope with
link objects.

### 7.4 Tables

```
GET /tables
POST /tables
GET /tables/{table_id}
PATCH /tables/{table_id}
DELETE /tables/{table_id}
GET /tables/{table_id}/rows
POST /tables/{table_id}/rows
PATCH /tables/{table_id}/rows/{row_id}
DELETE /tables/{table_id}/rows/{row_id}
```

Create table request:

```json
{
  "title": "Expenses",
  "description": "Personal spending",
  "columns": [
    {"name": "amount", "type": "number"},
    {"name": "category", "type": "select", "options": ["food", "travel"]}
  ],
  "parent_id": null,
  "tags": ["finance"]
}
```

Table response:

```json
{
  "id": "table-id",
  "item_id": "item-id",
  "title": "Expenses",
  "description": "Personal spending",
  "columns": [],
  "created_at": "2026-04-29T16:00:00Z",
  "updated_at": "2026-04-29T16:00:00Z"
}
```

`GET /tables/{table_id}` returns the table plus the first 20 rows:

```json
{
  "table": Table,
  "rows": [TableRow]
}
```

Rows list query parameters:

| Param | Type | Description |
|---|---|---|
| `filter` | JSON object string | exact-match filters by column |
| `order_by` | string | column name, prefix `-` for descending |
| `limit` | int | default 100, max 500 |
| `offset` | int | default 0 |

Row response:

```json
{
  "id": "row-id",
  "table_id": "table-id",
  "data": {"amount": 12.5, "category": "food"},
  "row_order": 0,
  "created_at": "2026-04-29T16:00:00Z",
  "updated_at": "2026-04-29T16:00:00Z"
}
```

Row create/update request:

```json
{
  "data": {"amount": 12.5, "category": "food"}
}
```

`GET /tables` and `GET /tables/{table_id}/rows` use the standard list envelope with
table and row objects in `items`.

`PATCH /tables/{table_id}` may update `title` and `description` only. V2 does not
support column schema mutation after table creation. Updating the table title also
updates the backing item's title.

`DELETE /tables/{table_id}` deletes table rows, the table record, and its backing
item atomically.

### 7.5 Links

```
POST /links
DELETE /links/{link_id}
```

Create request:

```json
{
  "source_id": "item-a",
  "target_id": "item-b",
  "link_type": "blocks"
}
```

Link response:

```json
{
  "id": "link-id",
  "source_id": "item-a",
  "target_id": "item-b",
  "link_type": "blocks",
  "created_at": "2026-04-29T16:00:00Z"
}
```

Duplicate link creation returns 409 with code `duplicate_link`.

### 7.6 Habit completions

```
GET /items/{item_id}/completions
POST /items/{item_id}/completions
DELETE /items/{item_id}/completions/{completion_id}
```

Create request:

```json
{
  "completed_on": "2026-04-29",
  "completed_at": "2026-04-29T16:00:00Z",
  "note": "felt easy today"
}
```

Response:

```json
{
  "id": "completion-id",
  "item_id": "habit-id",
  "completed_on": "2026-04-29",
  "completed_at": "2026-04-29T16:00:00Z",
  "note": "felt easy today",
  "created_at": "2026-04-29T16:00:00Z"
}
```

Duplicate completion for the same `completed_on` returns 409 with code
`duplicate_habit_completion`.

### 7.7 Auth tokens

```
POST /auth/apple
POST /auth/tokens
GET /auth/tokens
DELETE /auth/tokens/{token_id}
```

Create API key request:

```json
{"label": "local mcp"}
```

Create API key response:

```json
{
  "id": "key-id",
  "key": "bh_live_secret",
  "label": "local mcp",
  "created_at": "2026-04-29T16:00:00Z",
  "last_used_at": null
}
```

List response omits `key`.

### 7.8 Agent

```
POST /agent
GET /agent/brief
POST /agent/lint
```

Agent request:

```json
{
  "message": "what should I focus on today?"
}
```

Agent response:

```json
{
  "run_id": "agent-run-id",
  "response": "You have three important items...",
  "items_created": [Item],
  "items_updated": [Item],
  "tool_calls": [
    {
      "name": "list_items",
      "status": "success"
    }
  ]
}
```

`GET /agent/brief` response:

```json
{"response": "daily brief text"}
```

`POST /agent/lint` response:

```json
{
  "findings": [
    {
      "type": "stale_todo",
      "item_id": "item-id",
      "detail": "Open for more than 30 days."
    }
  ],
  "summary": "1 stale todo found."
}
```

### 7.9 Utility

```
GET /me
GET /tags
GET /epics
POST /search
```

`GET /me` response:

```json
{
  "id": "user-id",
  "item_counts": {
    "note": 10,
    "todo": 4
  }
}
```

`GET /tags` response:

```json
{"tags": ["finance", "product"]}
```

`GET /epics` is an alias for:

```
GET /items?type=epic&status=open
```

`POST /search` remains semantic search over items.

Request:

```json
{"query": "blackhole deployment"}
```

Response: list envelope of `Item`.

## 8. Agent tools

Agent tools are thin wrappers over the same domain service functions as HTTP. Tool
schemas use the same field names as HTTP schemas.

Model selection is configuration, not hardcoded in handlers:

```
OPENAI_FAST_MODEL=gpt-5.4-mini-2026-03-17
OPENAI_AGENT_MODEL=<configured-agent-model>
```

If `OPENAI_AGENT_MODEL` is unset, the backend may use the fast model during local
development, but production deployment must set it explicitly.

Production tools:

```
create_item       get_item        update_item
delete_item       list_items      search_items
create_table      get_table       query_table
upsert_row        delete_row
create_link       delete_link     get_item_links
get_context       get_daily_brief lint
```

Development-only tool:

```
run_python
```

`run_python` is registered only when:

```
BLACKHOLE_ENABLE_DEV_TOOLS=1
```

It must never be enabled in production Fly deployment.

### 8.1 Tool loop contract

1. Max tool turns: 8.
2. Per model call timeout: 45 seconds.
3. Per tool call timeout: 30 seconds.
4. Tool inputs and outputs are persisted to `agent_tool_calls`.
5. Tool exceptions are caught, logged, and returned as structured tool errors.
6. The final HTTP response includes created and updated items tracked by tool name.

### 8.2 Context snapshot

`get_context` and `/agent` use the same context builder:

```json
{
  "now": "2026-04-29T16:00:00Z",
  "active_epics": [Item],
  "open_todos": [Item],
  "todays_events": [Item],
  "overdue_todos": [Item]
}
```

The `/agent` endpoint stores this exact object in `agent_runs.context_json` before
the first model call.

## 9. iOS contract

### 9.1 Required model changes

`Item.swift` must decode both legacy and V2 item JSON.

Required fields:

```swift
let id: String
let content: String
let title: String
let type: ItemType
let status: ItemStatus
let priority: ItemPriority?
let source: String?
let parentId: String?
let epicId: String?
let dueDate: String?
var completed: Bool
let tags: [String]
let startTime: String?
let endTime: String?
let location: String?
let url: String?
let readStatus: String?
let email: String?
let phone: String?
let organization: String?
let recurrenceRule: [String: JSONValue]?
let metadata: [String: JSONValue]?
let createdAt: String
var updatedAt: String
```

Legacy decode defaults:

1. Missing `status` is derived from `completed`: `true -> done`, `false -> open`.
2. Missing `priority`, `source`, and `parentId` decode as nil.
3. Missing V2 optional type-specific fields decode as nil.
4. Missing `metadata` decodes as empty dictionary.

`ItemType` must support unknown future values without decode failure.

Example:

```swift
enum ItemType: Hashable, Codable {
    case note
    case todo
    case event
    case epic
    case contact
    case resource
    case decision
    case journal
    case habit
    case table
    case unknown(String)
}
```

### 9.2 UI changes

Keep:

1. Void tab voice capture.
2. Hold-to-dictate behavior.
3. Feed core list behavior.

Change:

1. Void tab submits to `POST /captures`.
2. Feed supports status filters instead of binary open/completed.
3. Item row shows priority indicator for `urgent` and `high`.
4. Add Ask tab for `POST /agent`.
5. Add Tables tab or nested Tables view for table listing and row editing.

Ask tab is single-turn and result-focused. It is not a persistent chat UI in V2.

## 10. MCP contract

MCP is not required to ship V2.

When implemented, MCP is a thin adapter over the HTTP API:

1. HTTP remains the source of truth.
2. MCP tools call HTTP or the same domain service contracts.
3. MCP requires API key auth.
4. External agents must call `get_context` before write tools.
5. MCP must not expose `run_python` unless dev tools are enabled.

## 11. Build sequence

### Phase 0 - iOS decode safety

1. Update iOS `Item` to decode V2 fields.
2. Add unknown item type handling.
3. Add iOS decoding tests for legacy JSON, V2 JSON, and unknown future type.

This phase should land before the backend emits new item types in production.

### Phase 1 - Data model and migration

1. Extend `init_db()` with additive migration guards for all new columns.
2. Create `tables`, `table_rows`, `item_links`, `habit_completions`, `api_keys`,
   `agent_runs`, and `agent_tool_calls`.
3. Implement domain validators and serializers.
4. Add DB CRUD functions needed by the domain service.
5. Add migration tests from the current schema.

### Phase 2 - Deterministic HTTP API

1. Add Pydantic schemas for all request/response bodies.
2. Add error envelope handling.
3. Add deterministic `POST /items`.
4. Add list filters and list envelopes.
5. Add table, row, link, habit completion, API key, and utility endpoints.
6. Extend auth middleware with API key support using token prefix dispatch.

### Phase 3 - Capture path

1. Add `POST /captures`.
2. Update fast classifier schema for the nine user-visible capture types.
3. Keep OpenAI Responses structured-output path.
4. Add legacy `POST /items {"content": ...}` alias for migration only.
5. Add fake-model classifier tests and opt-in live smoke tests.

### Phase 4 - Agent path

1. Implement OpenAI agent loop.
2. Implement production agent tools.
3. Add `agent_runs` and `agent_tool_calls` logging.
4. Add `/agent`, `/agent/brief`, and `/agent/lint`.
5. Add scripted fake-model agent loop tests.

### Phase 5 - iOS V2 UI

1. Update APIClient for `/captures`, V2 item fields, tables, and agent.
2. Add status filters and priority indicators.
3. Add Ask view.
4. Add Tables list/detail views.
5. Test against deployed/staging backend before enabling new capture types broadly.

### Phase 6 - Post-V2 candidates

1. MCP server.
2. Streaming agent responses.
3. Proactive linking.
4. Recurrence expansion and streak summaries.
5. Table schema migrations.
6. Admin link graph and habit charts.

## 12. Testing strategy

V2 is not complete without these tests.

### 12.1 Backend tests

Use real temporary SQLite databases. Do not mock `db.py` for contract tests.

Required coverage:

1. Migration from current schema preserves old rows.
2. Migration is idempotent.
3. Old `completed` rows produce correct `status`.
4. Old `epic_id` rows produce correct `parent_id`.
5. `GET /items` supports filters, pagination, and legacy fields.
6. `POST /items` deterministic create rejects invalid fields.
7. `PATCH /items` omitted/null semantics.
8. Every item type validates required fields.
9. Parent cycle rejection.
10. Epic parent rejection.
11. Event start/end validation.
12. Decision immutability.
13. Resource `read_status` validation.
14. Habit completion accepts only habits.
15. Duplicate habit completion returns 409.
16. Duplicate link returns 409.
17. Table column and row validation.
18. API key auth works and updates `last_used_at`.
19. Error responses use the standard envelope.

### 12.2 Classifier tests

Classifier tests use fake model outputs by default.

Required coverage:

1. Fake model output for each user-visible item type.
2. Invalid fake model output is rejected by the service layer.
3. Capture creates the minimum useful set of items.
4. Legacy `POST /items {"content": ...}` delegates to capture behavior.

Live OpenAI tests remain opt-in:

```bash
BLACKHOLE_RUN_OPENAI_TESTS=1 OPENAI_API_KEY=<key> python -m pytest tests/ -v -k smoke
```

### 12.3 Agent tests

Agent tests use a scripted fake model.

Required coverage:

1. No-tool response.
2. Successful item creation tool call.
3. Successful table query tool call.
4. Tool error returned to model and logged.
5. Max-turn failure.
6. Model timeout failure.
7. Tool timeout failure.
8. `agent_runs` status and context snapshot.
9. `agent_tool_calls` input/output/status persistence.

### 12.4 iOS tests

Required coverage:

1. Decode legacy item JSON.
2. Decode V2 item JSON.
3. Decode unknown future item type.
4. Encode capture request.
5. Decode capture response.
6. Decode agent response.
7. Decode table and row responses.

## 13. Rationale

### Why split `/captures` from `/items`?

Capture is an LLM workflow. Item creation is deterministic CRUD. Combining them
makes endpoint behavior depend on shape inference and makes testing ambiguous.

### Why keep OpenAI for V2?

The current backend already has OpenAI Responses infrastructure, logging, prompts,
and tests. V2 is already changing schema, API contracts, iOS models, and agent
tooling. Adding a provider migration at the same time increases risk without
unlocking the core V2 data model.

### Why make `status` canonical?

`completed` is too narrow for todos, events, resources, epics, and habits. Status
gives one lifecycle field across item types while preserving legacy iOS behavior
through a derived boolean.

### Why make `parent_id` canonical?

`epic_id` hardcodes one hierarchy. `parent_id` supports a generic tree while still
letting legacy clients see `epic_id` when the parent is an epic.

### Why store table backing items?

Tables need to participate in search, tags, links, and hierarchy. A backing item is
the simplest way to keep those concepts unified without duplicating graph features
for tables.

### Why defer MCP?

HTTP is the stable contract. MCP should be an adapter over that contract, not a
parallel source of truth.
