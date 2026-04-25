import os
from typing import Any

from openai import OpenAI

from . import tools

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def create_json_response(
    *,
    model: str,
    instructions: str,
    input_messages: list[dict[str, Any]],
    max_output_tokens: int,
    schema_name: str | None = None,
    schema: dict[str, Any] | None = None,
    tool_names: list[str] | tuple[str, ...] | None = None,
    openai_client: OpenAI | None = None,
) -> Any:
    text: dict[str, Any]
    if schema_name and schema:
        text = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        }
    else:
        text = {"format": {"type": "json_object"}}

    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_messages,
        "text": text,
        "max_output_tokens": max_output_tokens,
    }

    tool_definitions = tools.get_tool_definitions(tool_names)
    if tool_definitions:
        kwargs["tools"] = tool_definitions

    return (openai_client or get_client()).responses.create(**kwargs)


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return output_text

    parts: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)

    return "".join(parts)
