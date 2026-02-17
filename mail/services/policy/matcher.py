from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class MatchResult:
    category_id: str
    category_name: str
    confidence: float


def _score_category(email: dict, rules: Dict[str, Any]) -> int:
    score = 0
    sender_emails = {addr.get("email", "").lower() for addr in email.get("from", [])}
    subject = (email.get("subject") or "").lower()

    for sender in rules.get("sender_emails", []) or []:
        if sender.lower() in sender_emails:
            score += 5

    for domain in rules.get("sender_domains", []) or []:
        if any(addr.endswith("@" + domain.lower()) for addr in sender_emails):
            score += 3

    for keyword in rules.get("subject_keywords", []) or []:
        if keyword.lower() in subject:
            score += 2

    return score


def suggest_category(email: dict, categories: Iterable[dict]) -> Optional[MatchResult]:
    best: Optional[Tuple[int, dict]] = None
    for category in categories:
        rules = category.get("matching_rules") or {}
        score = _score_category(email, rules)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, category)

    if best is None:
        return None

    score, category = best
    confidence = min(1.0, 0.25 + (score / 10.0))
    return MatchResult(
        category_id=category["id"],
        category_name=category["name"],
        confidence=confidence,
    )
