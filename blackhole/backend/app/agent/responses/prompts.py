import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


ANALYSIS_MODEL = "gpt-5.4-mini-2026-03-17"

ANALYSIS_SYSTEM = (
    "You classify text submitted to blackhole and extract structured durable items. "
    "Valid item types are note, todo, and epic. "
    "The user's submitted text is the source of truth for each created item, so return "
    "the minimum useful set of items. For a note, memo, journal entry, meeting note, "
    "research dump, or document-like capture, usually create exactly one note with "
    "good title and tags; do not split sections into separate notes unless the user "
    "clearly asks for separate notes. Do not create a duplicate raw note just because "
    "you also created an epic or todo; only create the item types the user requested "
    "or clearly implied. Create todos for concrete actions the user intends to track. "
    "Create epics for high-level projects, outcomes, workstreams, or goals; the user "
    "may ask to create only an epic, and that should return one epic without an extra note. "
    "When the user's existing notes, todos, and epics are provided, use them as context "
    "for consistent tagging, detecting related items, and avoiding duplicates. "
    "Respond with valid JSON only, no prose."
)

ANALYSIS_PROMPT = """Classify this capture into the minimum set of durable items. Keep document-like notes as a single note unless the user explicitly asks for multiple separate notes. Split concrete action items into todos. Create an epic when the user names a project/workstream/goal or asks for an epic.

Transcript: {transcript}
Current UTC time: {now}

JSON response only:
{{
  "items": [
    {{
      "type": "note" or "todo" or "epic",
      "title": "concise summary (max 60 chars)",
      "tags": ["tag1"],
      "due_date": "ISO 8601 datetime or null (only for todos with an explicit date/time)"
    }}
  ]
}}

Return one item if the capture is a single thought, document, project, or requested epic. Return multiple only when there are clearly distinct durable items such as separate todos or explicitly separate notes."""

SEARCH_SYSTEM = "You find relevant items matching a search query. Respond with JSON only."

SEARCH_PROMPT = """Search query: "{query}"

Items (index: type | title | metadata | content preview):
{items_list}

Return indices of matching items, most relevant first (max 10). JSON only:
{{"indices": [0, 3, 1]}}"""

CONTEXT_INTRO = "Here are my recent notes, todos, and epics, newest first, for context:"
CONTEXT_ACK = "Got it. I'll use your existing items to classify the new transcript consistently."

CONTEXT_MAX_ITEMS = 25
CONTEXT_MAX_CHARS = 20_000

ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["note", "todo", "epic"]},
                    "title": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "due_date": {"type": ["string", "null"]},
                },
                "required": ["type", "title", "tags", "due_date"],
            },
        }
    },
    "required": ["items"],
}

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "indices": {
            "type": "array",
            "items": {"type": "integer"},
        }
    },
    "required": ["indices"],
}


@dataclass(frozen=True)
class ResponsePrompt:
    operation: str
    model: str
    instructions: str
    input_text: str
    input_messages: list[dict[str, Any]]
    max_output_tokens: int
    schema_name: str
    schema: dict[str, Any]
    tool_names: tuple[str, ...] = ()


def build_analysis_prompt(transcript: str, prior_items: list | None = None) -> ResponsePrompt:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    user_prompt = ANALYSIS_PROMPT.format(transcript=transcript, now=now)
    input_messages: list[dict[str, Any]] = []

    if prior_items:
        context_block = build_context_block(prior_items)
        if context_block:
            input_messages.append({"role": "user", "content": f"{CONTEXT_INTRO}\n\n{context_block}"})
            input_messages.append({"role": "assistant", "content": CONTEXT_ACK})

    input_messages.append({"role": "user", "content": user_prompt})

    return ResponsePrompt(
        operation="analyze_transcript",
        model=ANALYSIS_MODEL,
        instructions=ANALYSIS_SYSTEM,
        input_text=transcript,
        input_messages=input_messages,
        max_output_tokens=500,
        schema_name="transcript_items",
        schema=ITEMS_SCHEMA,
    )


def build_search_prompt(query: str, items: list) -> ResponsePrompt:
    items_list = "\n".join(_format_search_item(i, item) for i, item in enumerate(items))
    user_prompt = SEARCH_PROMPT.format(query=query, items_list=items_list)

    return ResponsePrompt(
        operation="search_items",
        model=ANALYSIS_MODEL,
        instructions=SEARCH_SYSTEM,
        input_text=query,
        input_messages=[{"role": "user", "content": user_prompt}],
        max_output_tokens=200,
        schema_name="search_indices",
        schema=SEARCH_SCHEMA,
    )


def _format_search_item(index: int, item: dict) -> str:
    tags = item.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

    item_type = item.get("type", "note")
    status = ""
    if item_type == "todo":
        status = "completed" if item.get("completed") else "open"
    due = f"due:{item['due_date']}" if item.get("due_date") else ""
    tag_str = ("tags:" + ",".join(tags)) if tags else ""
    metadata = " ".join(part for part in (status, due, tag_str) if part) or "none"
    title = item.get("title") or ""
    content = (item.get("content") or "")[:180]

    return f"{index}: {item_type} | {title} | {metadata} | {content}"


def build_context_block(items: list) -> str:
    lines = []
    total_chars = 0
    for item in items[:CONTEXT_MAX_ITEMS]:
        tags = item.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []

        item_type = item.get("type", "note")
        status = ""
        if item_type == "todo":
            status = " [DONE]" if item.get("completed") else " [OPEN]"

        due = f" due:{item['due_date']}" if item.get("due_date") else ""
        tag_str = (" #" + " #".join(tags)) if tags else ""
        content_preview = (item.get("content") or "")[:200]
        title = item.get("title") or ""

        line = f"[{item_type}{status}] {title} — {content_preview}{due}{tag_str}"
        if total_chars + len(line) > CONTEXT_MAX_CHARS:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)
