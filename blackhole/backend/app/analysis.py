import json
from typing import Any

from .agent.responses import client, logging, prompts

ANALYSIS_MODEL = prompts.ANALYSIS_MODEL


def get_client():
    return client.get_client()


def _run_prompt(prompt: prompts.ResponsePrompt) -> tuple[dict[str, Any] | None, dict]:
    raw_response = None
    try:
        response = client.create_json_response(
            model=prompt.model,
            instructions=prompt.instructions,
            input_messages=prompt.input_messages,
            max_output_tokens=prompt.max_output_tokens,
            schema_name=prompt.schema_name,
            schema=prompt.schema,
            tool_names=prompt.tool_names,
            openai_client=get_client(),
        )
        raw_response = client.response_text(response)
        parsed = json.loads(raw_response)
    except Exception as exc:
        return None, logging.build_log(
            operation=prompt.operation,
            model=prompt.model,
            input_text=prompt.input_text,
            system_prompt=prompt.instructions,
            input_messages=prompt.input_messages,
            raw_response=raw_response,
            parsed_response=None,
            status="error",
            error=str(exc),
        )

    return parsed, logging.build_log(
        operation=prompt.operation,
        model=prompt.model,
        input_text=prompt.input_text,
        system_prompt=prompt.instructions,
        input_messages=prompt.input_messages,
        raw_response=raw_response,
        parsed_response=parsed,
        status="success",
        error=None,
    )


def run_transcript_analysis(
    transcript: str, prior_items: list | None = None
) -> tuple[list[dict] | None, dict]:
    prompt = prompts.build_analysis_prompt(transcript, prior_items)
    parsed, log = _run_prompt(prompt)
    if parsed is None:
        return None, log

    if isinstance(parsed, list):
        items = parsed
    elif "items" in parsed and isinstance(parsed["items"], list):
        items = parsed["items"]
    else:
        items = [parsed]

    log["parsed_response"] = json.dumps(items)
    return items, log


def analyze_transcript(transcript: str, prior_items: list | None = None) -> list[dict]:
    results, log = run_transcript_analysis(transcript, prior_items)
    if results is None:
        raise RuntimeError(log["error"] or "Transcript analysis failed")
    return results


def run_search_items(query: str, items: list) -> tuple[list | None, dict]:
    prompt = prompts.build_search_prompt(query, items)
    parsed, log = _run_prompt(prompt)
    if parsed is None:
        return None, log

    indices = parsed.get("indices", []) if isinstance(parsed, dict) else []
    results = [items[i] for i in indices if isinstance(i, int) and 0 <= i < len(items)]
    log["parsed_response"] = json.dumps({"indices": indices})
    return results, log


def search_items(query: str, items: list) -> list:
    if not items:
        return []

    results, log = run_search_items(query, items)
    if results is None:
        raise RuntimeError(log["error"] or "Search analysis failed")
    return results
