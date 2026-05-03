import json
import os
import time
import uuid
from typing import Any

from .responses import client, logging, prompts
from .. import db, services

MAX_TOOL_TURNS = 8
MAX_TOOL_CALLS_PER_TURN = 4
AGENT_MAX_OUTPUT_TOKENS = 1200

TOOL_NAMES = [
    "get_context",
    "list_items",
    "search_items",
    "create_item",
    "update_item",
    "create_table",
    "query_table",
    "upsert_row",
    "create_link",
    "get_daily_brief",
    "lint",
]

AGENT_SYSTEM = """You are blackhole's backend agent.

You help the user inspect, organize, and update their personal blackhole data. You have tools for items, tables, links, daily brief, and lint findings. Use tools when you need current data or when the user asks you to create or update durable data. Do not invent item ids, table ids, tags, due dates, or completed state. Keep final answers concise and action-oriented.

Safety and data rules:
- Create durable items only when the user asks for something to be saved, tracked, planned, or organized.
- Prefer creating/updating through tools over asking the user to do manual follow-up.
- When the user asks you to create, add, remind, track, or plan a todo, call create_item with type "todo" in the first tool turn unless the request is missing the actual task.
- Never claim a mutation happened unless a tool result confirms it.
- Do not delete data; deletion is not available in this V2 agent loop.
- If tool output shows an error, explain the blocker and do not pretend it worked.

Response format — return JSON only, nothing else:
  While using tools:  {"response": null, "tool_calls": [{"name": "tool_name", "arguments": {"key": "value"}}]}
  When finished:      {"response": "your answer", "tool_calls": []}

Tool arguments are always a plain JSON object, never a string. Examples:
  create_item todo:  {"name": "create_item", "arguments": {"type": "todo", "title": "Pay rent"}}
  update status:     {"name": "update_item", "arguments": {"id": "abc123", "status": "done"}}
  list open todos:   {"name": "list_items", "arguments": {"type": "todo", "status": "open"}}

create_item accepts: type, title, content, status, priority, tags, parent_id, due_date, start_time, end_time, location, url, read_status, email, phone, organization, recurrence_rule, metadata.
update_item requires: id (plus any mutable fields to change).
list_items accepts: type, status, priority, parent_id, tags, query, limit, offset.
search_items accepts: query, limit, offset.
create_table accepts: title, description, columns, tags.
query_table accepts: table_id, filter, order_by, limit, offset.
upsert_row accepts: table_id, row_id (omit to insert), data.
create_link accepts: source_id, target_id, link_type.
get_context, get_daily_brief, and lint take {}."""


def run_agent(user_id: str, message: str) -> dict:
    context = build_context_snapshot(user_id)
    run_id = str(uuid.uuid4())
    model = _agent_model()
    created_at = services.now_iso()

    db.create_agent_run(
        {
            "id": run_id,
            "user_id": user_id,
            "message": message,
            "response": None,
            "context_json": json.dumps(context),
            "model": model,
            "status": "running",
            "error": None,
            "tool_turns": 0,
            "created_at": created_at,
            "completed_at": None,
        }
    )

    input_messages = [
        {
            "role": "user",
            "content": (
                "Context snapshot (json):\n"
                f"{json.dumps(_compact_for_prompt(context), ensure_ascii=False)}\n\n"
                f"User message:\n{message}"
            ),
        }
    ]
    created_items: dict[str, dict] = {}
    updated_items: dict[str, dict] = {}
    tool_summaries: list[dict] = []
    final_response: str | None = None

    try:
        for turn in range(MAX_TOOL_TURNS + 1):
            parsed = _model_step(
                model=model,
                input_messages=input_messages,
                user_id=user_id,
                run_id=run_id,
                turn=turn,
                message=message,
            )
            tool_calls = _normalize_tool_calls(parsed.get("tool_calls"))
            response_text = parsed.get("response")

            if not tool_calls:
                final_response = response_text or "Done."
                db.update_agent_run(
                    run_id,
                    {
                        "response": final_response,
                        "status": "success",
                        "tool_turns": turn,
                        "completed_at": services.now_iso(),
                    },
                )
                return _agent_response(run_id, final_response, created_items, updated_items, tool_summaries)

            if turn >= MAX_TOOL_TURNS:
                final_response = "I hit the tool-turn limit before finishing. Try a narrower request."
                db.update_agent_run(
                    run_id,
                    {
                        "response": final_response,
                        "status": "max_turns_exceeded",
                        "tool_turns": turn,
                        "completed_at": services.now_iso(),
                    },
                )
                return _agent_response(run_id, final_response, created_items, updated_items, tool_summaries)

            observations = []
            for index, tool_call in enumerate(tool_calls):
                result = _execute_and_log_tool(
                    run_id=run_id,
                    user_id=user_id,
                    turn=turn,
                    index=index,
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                )
                tool_summaries.append({"name": tool_call["name"], "status": result["status"]})
                observations.append(result)
                _track_item_changes(user_id, result, created_items, updated_items)

            input_messages.append({"role": "assistant", "content": json.dumps(parsed)})
            input_messages.append({"role": "user", "content": "Tool results:\n" + json.dumps(_compact_for_prompt(observations), ensure_ascii=False)})
    except Exception as exc:
        final_response = "I couldn't complete that agent turn."
        db.update_agent_run(
            run_id,
            {
                "response": final_response,
                "status": "error",
                "error": str(exc),
                "tool_turns": len(tool_summaries),
                "completed_at": services.now_iso(),
            },
        )
        return _agent_response(run_id, final_response, created_items, updated_items, tool_summaries)

    final_response = "I hit the tool-turn limit before finishing. Try a narrower request."
    db.update_agent_run(
        run_id,
        {
            "response": final_response,
            "status": "max_turns_exceeded",
            "tool_turns": MAX_TOOL_TURNS,
            "completed_at": services.now_iso(),
        },
    )
    return _agent_response(run_id, final_response, created_items, updated_items, tool_summaries)


def build_context_snapshot(user_id: str) -> dict:
    recent = [services.row_to_item(row) for row in db.list_items_recent(user_id, 25)]
    open_todos = services.list_items(user_id, item_type="todo", status="open", limit=25)["items"]
    in_progress = services.list_items(user_id, status="in_progress", limit=25)["items"]
    upcoming_events = services.list_items(user_id, item_type="event", limit=25)["items"]
    tables = services.list_tables(user_id, limit=25)["items"]
    tags = db.list_tags(user_id)
    lint = services.lint_items(user_id)
    return {
        "captured_at": services.now_iso(),
        "recent_items": recent,
        "open_todos": open_todos,
        "in_progress": in_progress,
        "upcoming_events": upcoming_events,
        "tables": tables,
        "tags": tags,
        "lint": lint,
    }


def _model_step(
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    user_id: str | None = None,
    run_id: str | None = None,
    turn: int | None = None,
    message: str | None = None,
) -> dict:
    raw = None
    parsed = None
    try:
        response = client.create_json_response(
            model=model,
            instructions=AGENT_SYSTEM,
            input_messages=input_messages,
            max_output_tokens=AGENT_MAX_OUTPUT_TOKENS,
        )
        raw = client.response_text(response)
        parsed = _parse_model_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Agent model response must be a JSON object.")
    except Exception as exc:
        _persist_agent_llm_log(
            user_id=user_id,
            run_id=run_id,
            turn=turn,
            model=model,
            message=message,
            input_messages=input_messages,
            raw_response=raw,
            parsed_response=parsed,
            status="error",
            error=str(exc),
        )
        raise

    _persist_agent_llm_log(
        user_id=user_id,
        run_id=run_id,
        turn=turn,
        model=model,
        message=message,
        input_messages=input_messages,
        raw_response=raw,
        parsed_response=parsed,
        status="success",
        error=None,
    )
    return parsed


def _execute_and_log_tool(
    *,
    run_id: str,
    user_id: str,
    turn: int,
    index: int,
    name: str,
    arguments: dict[str, Any],
) -> dict:
    start = time.monotonic()
    status = "success"
    error = None
    output: Any = None
    try:
        output = execute_tool(user_id, name, arguments)
    except Exception as exc:
        status = "error"
        error = str(exc)
        output = {"error": error}

    duration_ms = int((time.monotonic() - start) * 1000)
    db.create_agent_tool_call(
        {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "user_id": user_id,
            "tool_call_id": f"{turn}:{index}",
            "name": name,
            "input_json": json.dumps(arguments),
            "output_json": json.dumps(output),
            "status": status,
            "error": error,
            "duration_ms": duration_ms,
            "created_at": services.now_iso(),
        }
    )
    return {"name": name, "arguments": arguments, "status": status, "output": output}


def execute_tool(user_id: str, name: str, arguments: dict[str, Any]) -> Any:
    args = arguments or {}
    if name == "get_context":
        return build_context_snapshot(user_id)
    if name == "list_items":
        return services.list_items(
            user_id,
            item_type=args.get("type"),
            status=args.get("status"),
            priority=args.get("priority"),
            parent_id=args.get("parent_id"),
            tags=args.get("tags"),
            query=args.get("query"),
            limit=args.get("limit", 25),
            offset=args.get("offset", 0),
        )
    if name == "search_items":
        return services.list_items(
            user_id,
            query=args.get("query"),
            limit=args.get("limit", 25),
            offset=args.get("offset", 0),
        )
    if name == "create_item":
        payload = dict(args)
        return services.create_item(user_id, payload, default_source="agent")
    if name == "update_item":
        item_id = args.get("id") or args.get("item_id")
        if not item_id:
            raise services.ServiceError("item_id_required", "update_item requires id.")
        payload = {key: value for key, value in args.items() if key not in {"id", "item_id"}}
        return services.update_item(user_id, item_id, payload)
    if name == "create_table":
        return services.create_table(user_id, args)
    if name == "query_table":
        table_id = args.get("table_id")
        if not table_id:
            raise services.ServiceError("table_id_required", "query_table requires table_id.")
        return services.list_table_rows(
            user_id,
            table_id,
            filter_json=json.dumps(args["filter"]) if isinstance(args.get("filter"), dict) else args.get("filter"),
            order_by=args.get("order_by"),
            limit=args.get("limit", 25),
            offset=args.get("offset", 0),
        )
    if name == "upsert_row":
        table_id = args.get("table_id")
        if not table_id:
            raise services.ServiceError("table_id_required", "upsert_row requires table_id.")
        payload = {"data": args.get("data") or {}}
        row_id = args.get("row_id") or args.get("id")
        if row_id:
            return services.update_table_row(user_id, table_id, row_id, payload)
        return services.create_table_row(user_id, table_id, payload)
    if name == "create_link":
        return services.create_link(user_id, args)
    if name == "get_daily_brief":
        return services.get_agent_brief(user_id)
    if name == "lint":
        return services.lint_items(user_id)
    raise services.ServiceError("unknown_agent_tool", f"Unknown agent tool: {name}")


def _normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tool_calls = []
    for call in value[:MAX_TOOL_CALLS_PER_TURN]:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        # Accept arguments as a plain object OR as a legacy JSON-encoded string
        arguments = _parse_tool_arguments(call.get("arguments", call.get("arguments_json")))
        if name in TOOL_NAMES and isinstance(arguments, dict):
            tool_calls.append({"name": name, "arguments": arguments})
    return tool_calls


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_model_json(raw: str) -> Any:
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(raw.lstrip())
    return parsed


def _persist_agent_llm_log(
    *,
    user_id: str | None,
    run_id: str | None,
    turn: int | None,
    model: str,
    message: str | None,
    input_messages: list[dict[str, Any]],
    raw_response: str | None,
    parsed_response: Any,
    status: str,
    error: str | None,
) -> None:
    if not user_id:
        return

    log = logging.build_log(
        operation="agent_turn",
        model=model,
        input_text=message or "",
        system_prompt=AGENT_SYSTEM,
        input_messages=input_messages,
        raw_response=raw_response,
        parsed_response=parsed_response,
        status=status,
        error=error,
    )
    log.update(
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "item_id": run_id,
            "created_at": services.now_iso(),
        }
    )

    try:
        db.create_llm_log(log)
    except Exception:
        pass


def _track_item_changes(
    user_id: str,
    result: dict[str, Any],
    created_items: dict[str, dict],
    updated_items: dict[str, dict],
) -> None:
    if result["status"] != "success" or not isinstance(result.get("output"), dict):
        return
    output = result["output"]
    name = result["name"]
    if name == "create_item" and output.get("id"):
        created_items[output["id"]] = output
    if name == "update_item" and output.get("id"):
        updated_items[output["id"]] = output
    if name == "create_table" and output.get("item_id"):
        try:
            item = services.get_item(user_id, output["item_id"])
        except Exception:
            item = None
        if item:
            created_items[item["id"]] = item


def _agent_response(
    run_id: str,
    response: str,
    created_items: dict[str, dict],
    updated_items: dict[str, dict],
    tool_summaries: list[dict],
) -> dict:
    return {
        "run_id": run_id,
        "response": response,
        "items_created": list(created_items.values()),
        "items_updated": list(updated_items.values()),
        "tool_calls": tool_summaries,
    }


def _compact_for_prompt(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= 18_000:
        return value
    return {"truncated": True, "preview": text[:18_000]}


def _agent_model() -> str:
    return (
        os.getenv("OPENAI_AGENT_MODEL")
        or os.getenv("OPENAI_FAST_MODEL")
        or prompts.ANALYSIS_MODEL
    )
