import json
from typing import Any


def prompt_payload(input_messages: list[dict[str, Any]]) -> str:
    return json.dumps(input_messages)


def build_log(
    *,
    operation: str,
    model: str,
    input_text: str,
    system_prompt: str,
    input_messages: list[dict[str, Any]],
    raw_response: str | None,
    parsed_response: Any,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "model": model,
        "input_text": input_text,
        "system_prompt": system_prompt,
        "user_prompt": prompt_payload(input_messages),
        "raw_response": raw_response,
        "parsed_response": json.dumps(parsed_response) if parsed_response is not None else None,
        "status": status,
        "error": error,
    }
