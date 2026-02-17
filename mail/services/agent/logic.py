from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services.policy.enforcer import policy_allows_send
from services.policy.matcher import suggest_category


def _sender_name(email: dict) -> str:
    from_list = email.get("from") or []
    if from_list:
        name = from_list[0].get("name")
        email_addr = from_list[0].get("email")
        return name or (email_addr.split("@")[0] if email_addr else "")
    return ""


def generate_draft(
    email: dict,
    instruction: str,
    policy: Optional[Dict[str, Any]],
    signature: str,
) -> Dict[str, Any]:
    sender = _sender_name(email)
    greeting = f"Hi {sender}," if sender else "Hi,"
    subject = email.get("subject") or ""

    template = (policy or {}).get("reply_template")
    if template:
        try:
            body = template.format(
                instruction=instruction,
                sender=sender or "there",
                subject=subject,
            )
        except (KeyError, ValueError):
            body = f"{template}\n\n{instruction.strip()}"
    else:
        body = instruction.strip()

    draft = f"{greeting}\n\n{body}\n\n{signature}\n"

    allowed, reason = policy_allows_send(policy or {}, subject, draft)
    checks = [f"policy:{'pass' if allowed else 'fail'}:{reason}"]

    return {
        "primary_artifact": draft,
        "rationale_short": "Draft uses the instruction and the sender context.",
        "citations": [email.get("id")],
        "confidence": 0.5,
        "policy_checks": checks,
    }


def suggest_snooze(email: dict) -> Dict[str, Any]:
    subject = (email.get("subject") or "").lower()
    days = 2
    reason = "follow_up"

    if "newsletter" in subject or "fyi" in subject:
        days = 7
        reason = "newsletter"
    elif "urgent" in subject or "asap" in subject:
        days = 1
        reason = "urgent"

    until = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

    return {
        "primary_artifact": {"until": until, "reason": reason},
        "rationale_short": "Heuristic based on subject urgency.",
        "citations": [email.get("id")],
        "confidence": 0.4,
        "policy_checks": [],
    }


def classify_email(email: dict, categories: list[dict]) -> Dict[str, Any]:
    match = suggest_category(email, categories)
    suggestion = None
    if match:
        suggestion = {
            "category_id": match.category_id,
            "category_name": match.category_name,
            "confidence": match.confidence,
        }

    return {
        "primary_artifact": suggestion,
        "rationale_short": "Matched against deterministic category rules.",
        "citations": [email.get("id")],
        "confidence": suggestion.get("confidence") if suggestion else 0.1,
        "policy_checks": [],
    }


def build_notes_markdown(items: list[dict]) -> str:
    lines = ["# Remembered", "", "## Recent"]
    for item in items:
        created = item.get("created_at")
        summary = item.get("summary")
        email_id = item.get("email_id")
        stamp = created.isoformat() if hasattr(created, "isoformat") else str(created)
        lines.append(f"- {stamp}: {summary} ({email_id})")
    lines.append("")
    return "\n".join(lines)
