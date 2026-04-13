import json
import os
from datetime import datetime

from openai import OpenAI

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


_ANALYSIS_SYSTEM = "You classify voice transcripts and extract structured data. Respond with valid JSON only, no prose."
ANALYSIS_MODEL = "gpt-5.4-mini-2026-03-17"

_ANALYSIS_PROMPT = """Classify this voice transcript.

Transcript: {transcript}
Current UTC time: {now}

JSON response only:
{{
  "type": "note" or "todo",
  "title": "concise summary (max 60 chars)",
  "tags": ["tag1"],
  "due_date": "ISO 8601 datetime or null (only for todos if a date is mentioned)"
}}"""


def run_transcript_analysis(transcript: str) -> tuple[dict | None, dict]:
    client = get_client()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    user_prompt = _ANALYSIS_PROMPT.format(transcript=transcript, now=now)
    raw_response = None

    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _ANALYSIS_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
        )
        raw_response = response.choices[0].message.content
        parsed = json.loads(raw_response)
    except Exception as exc:
        return None, {
            "operation": "analyze_transcript",
            "model": ANALYSIS_MODEL,
            "input_text": transcript,
            "system_prompt": _ANALYSIS_SYSTEM,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_response": None,
            "status": "error",
            "error": str(exc),
        }

    return parsed, {
        "operation": "analyze_transcript",
        "model": ANALYSIS_MODEL,
        "input_text": transcript,
        "system_prompt": _ANALYSIS_SYSTEM,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "parsed_response": json.dumps(parsed),
        "status": "success",
        "error": None,
    }


def analyze_transcript(transcript: str) -> dict:
    result, log = run_transcript_analysis(transcript)
    if result is None:
        raise RuntimeError(log["error"] or "Transcript analysis failed")
    return result


_SEARCH_SYSTEM = "You find relevant items matching a search query. Respond with JSON only."

_SEARCH_PROMPT = """Search query: "{query}"

Items (index: title | content preview):
{items_list}

Return indices of matching items, most relevant first (max 10). JSON only:
{{"indices": [0, 3, 1]}}"""


def search_items(query: str, items: list) -> list:
    if not items:
        return []

    client = get_client()
    items_list = "\n".join(
        f"{i}: {item['title']} | {item['content'][:150]}"
        for i, item in enumerate(items)
    )

    response = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SEARCH_SYSTEM},
            {"role": "user", "content": _SEARCH_PROMPT.format(query=query, items_list=items_list)},
        ],
        max_tokens=200,
    )

    result = json.loads(response.choices[0].message.content)
    indices = result.get("indices", [])
    return [items[i] for i in indices if isinstance(i, int) and 0 <= i < len(items)]
