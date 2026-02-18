from __future__ import annotations

import re

TAG_CATALOG: list[str] = [
    "anxious",
    "stressed",
    "overwhelmed",
    "sad",
    "lonely",
    "angry",
    "frustrated",
    "irritable",
    "guilty",
    "ashamed",
    "numb",
    "tired",
    "burned out",
    "restless",
    "scared",
    "hopeful",
    "content",
    "calm",
    "grateful",
    "joyful",
    "motivated",
    "focused",
    "confident",
    "proud",
    "relieved",
    "curious",
    "work pressure",
    "school pressure",
    "relationship tension",
    "family stress",
    "friendship",
    "social anxiety",
    "health worry",
    "sleep issues",
    "financial stress",
    "self doubt",
    "perfectionism",
    "procrastination",
    "decision fatigue",
    "major transition",
    "grief",
    "conflict",
    "growth",
    "self care",
    "exercise",
    "creativity",
    "purpose",
    "isolation",
    "connection",
    "reflection",
]

TAG_SET = set(TAG_CATALOG)

TAG_ALIASES: dict[str, str] = {
    "burnout": "burned out",
    "burnedout": "burned out",
    "burnt out": "burned out",
    "family": "family stress",
    "relationship": "relationship tension",
    "relationships": "relationship tension",
    "work": "work pressure",
    "school": "school pressure",
    "money stress": "financial stress",
    "money": "financial stress",
    "health": "health worry",
    "sleep": "sleep issues",
    "self-care": "self care",
}

TAG_KEYWORDS: dict[str, set[str]] = {
    "anxious": {"anxious", "nervous", "panic", "worry", "worried"},
    "stressed": {"stressed", "stress", "under pressure"},
    "overwhelmed": {"overwhelmed", "too much", "swamped"},
    "sad": {"sad", "down", "upset"},
    "lonely": {"lonely", "alone", "isolated"},
    "angry": {"angry", "mad", "furious"},
    "frustrated": {"frustrated", "annoyed", "fed up"},
    "irritable": {"irritable", "snappy", "on edge"},
    "guilty": {"guilty", "regret"},
    "ashamed": {"ashamed", "embarrassed"},
    "numb": {"numb", "detached", "empty"},
    "tired": {"tired", "drained", "exhausted"},
    "burned out": {"burned out", "burnout"},
    "restless": {"restless", "can't settle", "cant settle"},
    "scared": {"scared", "afraid", "fear"},
    "hopeful": {"hopeful", "optimistic"},
    "content": {"content", "okay", "at ease"},
    "calm": {"calm", "peaceful", "steady"},
    "grateful": {"grateful", "thankful", "appreciate"},
    "joyful": {"joyful", "happy", "glad"},
    "motivated": {"motivated", "driven", "ready"},
    "focused": {"focused", "clear", "dialed in"},
    "confident": {"confident", "assured", "certain"},
    "proud": {"proud", "accomplished"},
    "relieved": {"relieved", "lighter"},
    "curious": {"curious", "interested", "wondering"},
    "work pressure": {"work", "job", "deadline", "manager", "meeting"},
    "school pressure": {"school", "class", "exam", "homework"},
    "relationship tension": {"partner", "relationship", "argument", "fight"},
    "family stress": {"family", "parents", "sibling"},
    "friendship": {"friend", "friends"},
    "social anxiety": {"social anxiety", "social", "awkward around people"},
    "health worry": {"health", "symptom", "sick", "ill"},
    "sleep issues": {"sleep", "insomnia", "couldn't sleep", "couldnt sleep"},
    "financial stress": {"money", "rent", "bills", "debt", "financial"},
    "self doubt": {"self doubt", "not enough", "insecure", "doubt myself"},
    "perfectionism": {"perfection", "perfect", "high standards"},
    "procrastination": {"procrastinating", "procrastination", "put off"},
    "decision fatigue": {"decision fatigue", "too many decisions", "can't decide", "cant decide"},
    "major transition": {"transition", "new chapter", "change"},
    "grief": {"grief", "mourning", "loss"},
    "conflict": {"conflict", "tension", "clash"},
    "growth": {"growth", "learning", "progress"},
    "self care": {"self care", "self-care", "resting"},
    "exercise": {"exercise", "workout", "run", "gym"},
    "creativity": {"creative", "creativity", "art", "writing"},
    "purpose": {"purpose", "meaning", "direction"},
    "isolation": {"isolation", "withdrawn"},
    "connection": {"connection", "connected", "supported"},
    "reflection": {"reflecting", "reflection", "processing"},
}

THEME_KEYWORDS: dict[str, set[str]] = {
    "work": {"work", "job", "meeting", "manager", "deadline"},
    "relationships": {"partner", "friend", "family", "relationship", "parents"},
    "health": {"sleep", "exercise", "health", "sick", "tired"},
    "self-worth": {"confidence", "worth", "failure", "enough", "self"},
}


def contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _normalize_tag_value(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def normalize_tag(value: str) -> str | None:
    normalized = _normalize_tag_value(value)
    if not normalized:
        return None
    if normalized in TAG_SET:
        return normalized
    alias = TAG_ALIASES.get(normalized)
    if alias in TAG_SET:
        return alias
    return None


def clean_tags(raw_tags: list[str], minimum: int = 1, maximum: int = 4) -> list[str]:
    safe_minimum = max(1, minimum)
    safe_maximum = max(safe_minimum, min(4, maximum))

    cleaned: list[str] = []
    for raw in raw_tags:
        normalized = normalize_tag(raw)
        if normalized is None:
            continue
        if normalized in cleaned:
            continue
        cleaned.append(normalized)
        if len(cleaned) >= safe_maximum:
            break

    if len(cleaned) < safe_minimum:
        return ["reflection"]
    return cleaned


def extract_mood_tags(lowered_text: str, limit: int = 4) -> list[str]:
    safe_limit = max(1, min(4, limit))
    scores: list[tuple[str, int, int]] = []
    for index, tag in enumerate(TAG_CATALOG):
        words = TAG_KEYWORDS.get(tag, set())
        count = sum(1 for word in words if contains_phrase(lowered_text, word))
        if count > 0:
            scores.append((tag, count, index))

    scores.sort(key=lambda item: (-item[1], item[2]))
    tags = [tag for tag, _, _ in scores[:safe_limit]]
    return tags or ["reflection"]


def extract_themes(lowered_text: str, limit: int = 4) -> list[str]:
    themes: list[str] = []
    for theme, words in THEME_KEYWORDS.items():
        if any(contains_phrase(lowered_text, word) for word in words):
            themes.append(theme)
    return themes[:limit]
