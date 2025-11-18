from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1-mini")
API_KEY = os.getenv("OPENAI_API_KEY")
AI_ENABLED = bool(API_KEY)
client: Optional[OpenAI] = OpenAI(api_key=API_KEY) if AI_ENABLED else None

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "note_enrichment",
        "schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short kebab-case topic tags for the note.",
                },
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence description of the note referencing linked context where relevant.",
                },
            },
            "required": ["tags", "summary"],
            "additionalProperties": False,
        },
    },
}


def generate_ai_metadata(
    content: str,
    manual_tags: List[str],
    existing_tags: List[str],
    linked_notes: List[Dict[str, str]],
) -> Optional[Dict[str, object]]:
    if not client:
        return None

    linked_blurbs = "\n".join(
        f"- {item['summary']} (tags: {', '.join(item['tags'])})" for item in linked_notes
    )
    if not linked_blurbs:
        linked_blurbs = "- None yet — this might be a seed insight."

    manual = ", ".join(manual_tags) if manual_tags else "None supplied"
    catalog = ", ".join(existing_tags) if existing_tags else "No tags exist yet"

    prompt = f"""
You are the research librarian for an AI lab. Given a raw note, existing tag catalogue, manual tags, and
linked note snippets, output JSON that contains (1) new tags (reuse existing ones when appropriate, otherwise create
succinct new ones) and (2) a crisp description tying this note into the linked work.

Existing tags: {catalog}
Manual tags to force-include: {manual}
Linked notes:\n{linked_blurbs}

Note:
\"\"\"{content.strip()}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            response_format=SCHEMA,
            messages=[
                {
                    "role": "system",
                    "content": "Respond only with JSON following the schema. Tags should be lowercase kebab-case.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        message = response.choices[0].message.content
        if not message:
            return None
        return json.loads(message)
    except Exception:
        return None
