from __future__ import annotations

from typing import Optional

from openai import OpenAI
from openai.types.responses import Response

QORK_SYSTEM_PROMPT = (
    "You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. "
    "You should be concise and to the point. Prioritize answers, and explanations ONLY when requested."
)


def get_response(
    *,
    model: str,
    prompt: str,
    api_key: str,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[dict] = None,
) -> Response:
    client = OpenAI(api_key=api_key)
    kwargs = {}
    if reasoning:
        kwargs["reasoning"] = reasoning
    return client.responses.create(
        model=model,
        input=prompt,
        instructions=QORK_SYSTEM_PROMPT,
        previous_response_id=previous_response_id,
        **kwargs,
    )


def stream_response(
    *,
    model: str,
    prompt: str,
    api_key: str,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[dict] = None,
):
    """Return a ResponseStreamManager context manager (see OpenAI SDK)."""
    client = OpenAI(api_key=api_key)
    kwargs = {}
    if reasoning:
        kwargs["reasoning"] = reasoning
    return client.responses.stream(
        model=model,
        input=prompt,
        instructions=QORK_SYSTEM_PROMPT,
        previous_response_id=previous_response_id,
        **kwargs,
    )


def response_text(response: Response) -> str:
    try:
        text = getattr(response, "output_text", None)
    except Exception:
        text = None
    if text:
        return text
    try:
        return str(response)
    except Exception:
        return "(No content)"
