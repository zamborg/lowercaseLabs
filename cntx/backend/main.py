from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .ai import AI_ENABLED, generate_ai_metadata
except ImportError:  # pragma: no cover
    from ai import AI_ENABLED, generate_ai_metadata


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "cntx.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                tags TEXT NOT NULL,
                link_summary TEXT NOT NULL,
                embedding TEXT NOT NULL,
                manual_tags TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
        if "manual_tags" not in columns:
            conn.execute("ALTER TABLE notes ADD COLUMN manual_tags TEXT NOT NULL DEFAULT '[]'")


init_db()


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "we",
    "our",
    "from",
    "can",
    "not",
    "have",
    "has",
    "had",
    "were",
    "also",
    "about",
    "more",
    "less",
    "than",
    "when",
    "while",
    "where",
    "who",
    "how",
    "what",
}

TOKEN_SANITIZER = re.compile(r"[^a-z0-9\s]+")


def normalize_tags(tags: Optional[List[str]]) -> List[str]:
    if not tags:
        return []
    cleaned: List[str] = []
    seen: Set[str] = set()
    for tag in tags:
        token = TOKEN_SANITIZER.sub(" ", tag.lower()).strip()
        token = re.sub(r"\s+", "-", token)
        if token and token not in seen:
            seen.add(token)
            cleaned.append(token)
    return cleaned


class NoteBase(BaseModel):
    content: str = Field(..., min_length=1)
    manualTags: List[str] = Field(default_factory=list)
    useAi: bool = False


class NoteCreate(NoteBase):
    pass


class NoteUpdate(BaseModel):
    content: Optional[str] = None
    manualTags: Optional[List[str]] = None
    useAi: Optional[bool] = None


class NoteOut(BaseModel):
    id: str
    content: str
    createdAt: str
    autoTags: List[str]
    manualTags: List[str]
    tags: List[str]
    linkSummary: str
    score: Optional[float] = None


class TagOut(BaseModel):
    name: str
    noteCount: int


def serialize_note(note: Dict[str, object]) -> Dict[str, object]:
    payload = {
        "id": note["id"],
        "content": note["content"],
        "createdAt": note["createdAt"],
        "autoTags": note["autoTags"],
        "manualTags": note["manualTags"],
        "tags": note["tags"],
        "linkSummary": note["linkSummary"],
    }
    if "score" in note:
        payload["score"] = note["score"]
    return payload

app = FastAPI(title="CNTX API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/api/notes")
def list_notes(q: Optional[str] = None) -> Dict[str, object]:
    notes = fetch_notes()
    if not q:
        return {"notes": [serialize_note(note) for note in notes]}

    query_embedding = embed(q)
    scored = []
    for note in notes:
        score = cosine_similarity(query_embedding, note["embedding"])
        if score > 0.05:
            scored.append({**note, "score": score})
    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return {"notes": [serialize_note(note) for note in scored], "query": q}


@app.get("/api/notes/{note_id}")
def get_note(note_id: str) -> Dict[str, NoteOut]:
    note = fetch_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"note": serialize_note(note)}


@app.post("/api/notes")
def create_note(payload: NoteCreate) -> Dict[str, NoteOut]:
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    manual_tags = normalize_tags(payload.manualTags)
    embedding = embed(content)
    existing_notes = fetch_notes()
    existing_tags = collect_existing_tags(existing_notes)
    neighbors = build_neighbor_context(embedding, existing_notes)

    auto_tags = generate_tags(content)
    link_summary = build_link_summary(embedding, auto_tags + manual_tags, existing_notes)

    if payload.useAi and AI_ENABLED:
        ai_result = generate_ai_metadata(content, manual_tags, existing_tags, neighbors)
        if ai_result:
            ai_tags = normalize_tags(ai_result.get("tags", []))
            if ai_tags:
                auto_tags = ai_tags
            ai_summary = ai_result.get("summary")
            if ai_summary:
                link_summary = ai_summary

    note = {
        "id": uuid4().hex,
        "content": content,
        "autoTags": auto_tags,
        "manualTags": manual_tags,
        "tags": sorted({*auto_tags, *manual_tags}),
        "embedding": embedding,
        "linkSummary": link_summary,
        "createdAt": datetime.utcnow().isoformat(),
    }
    persist_note(note)
    stored = fetch_note(note["id"])
    return {"note": serialize_note(stored or note)}


@app.put("/api/notes/{note_id}")
def update_note(note_id: str, payload: NoteUpdate) -> Dict[str, NoteOut]:
    existing = fetch_note(note_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Note not found")

    content = payload.content.strip() if payload.content is not None else existing["content"]
    if not content:
        raise HTTPException(status_code=400, detail="content cannot be empty")

    manual_tags = (
        normalize_tags(payload.manualTags) if payload.manualTags is not None else existing["manualTags"]
    )

    auto_tags = existing["autoTags"]
    embedding = existing["embedding"]
    if payload.content is not None:
        auto_tags = generate_tags(content)
        embedding = embed(content)

    other_notes = [note for note in fetch_notes() if note["id"] != note_id]
    existing_tags = collect_existing_tags(other_notes)
    neighbors = build_neighbor_context(embedding, other_notes)

    link_summary = build_link_summary(embedding, auto_tags + manual_tags, other_notes)
    use_ai = bool(payload.useAi) and AI_ENABLED
    if use_ai:
        ai_result = generate_ai_metadata(content, manual_tags, existing_tags, neighbors)
        if ai_result:
            ai_tags = normalize_tags(ai_result.get("tags", []))
            if ai_tags:
                auto_tags = ai_tags
            ai_summary = ai_result.get("summary")
            if ai_summary:
                link_summary = ai_summary

    updated = {
        **existing,
        "content": content,
        "autoTags": auto_tags,
        "manualTags": manual_tags,
        "tags": sorted({*auto_tags, *manual_tags}),
        "embedding": embedding,
        "linkSummary": link_summary,
        "createdAt": existing["createdAt"],
    }
    persist_note(updated)
    stored = fetch_note(note_id)
    return {"note": serialize_note(stored or updated)}


@app.get("/api/graph")
def graph() -> Dict[str, object]:
    notes = fetch_notes()
    return build_graphs(notes)


@app.get("/api/tags")
def list_tags() -> Dict[str, List[TagOut]]:
    notes = fetch_notes()
    counter = Counter()
    for note in notes:
        for tag in note["tags"]:
            counter[tag] += 1
    tags = [{"name": name, "noteCount": count} for name, count in counter.most_common()]
    return {"tags": tags}


def fetch_notes() -> List[Dict[str, object]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM notes ORDER BY datetime(created_at) DESC").fetchall()
    return [row_to_note(row) for row in rows]


def fetch_note(note_id: str) -> Optional[Dict[str, object]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    return row_to_note(row) if row else None


def persist_note(note: Dict[str, object]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO notes (id, content, created_at, tags, link_summary, embedding, manual_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,
                created_at=excluded.created_at,
                tags=excluded.tags,
                link_summary=excluded.link_summary,
                embedding=excluded.embedding,
                manual_tags=excluded.manual_tags
            """,
            (
                note["id"],
                note["content"],
                note["createdAt"],
                json.dumps(note["autoTags"]),
                note["linkSummary"],
                json.dumps(note["embedding"]),
                json.dumps(note["manualTags"]),
            ),
        )


def row_to_note(row: sqlite3.Row) -> Dict[str, object]:
    auto_tags = json.loads(row["tags"])
    manual_raw = row["manual_tags"] if "manual_tags" in row.keys() else "[]"
    manual_tags = json.loads(manual_raw) if manual_raw else []
    combined = sorted({*auto_tags, *manual_tags})
    return {
        "id": row["id"],
        "content": row["content"],
        "createdAt": row["created_at"],
        "autoTags": auto_tags,
        "manualTags": manual_tags,
        "tags": combined,
        "linkSummary": row["link_summary"],
        "embedding": json.loads(row["embedding"]),
    }


def tokenize(text: str) -> List[str]:
    clean = TOKEN_SANITIZER.sub(" ", text.lower())
    return [token for token in clean.split() if token]


def hash_token(token: str, dimensions: int) -> int:
    result = 0
    for char in token:
        result = (result << 5) - result + ord(char)
        result &= 0xFFFFFFFF
    return abs(result) % dimensions


def embed(text: str, dimensions: int = 128) -> List[float]:
    tokens = tokenize(text)
    vector = [0.0] * dimensions
    for token in tokens:
        index = hash_token(token, dimensions)
        vector[index] += 1.0
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def generate_tags(text: str, limit: int = 5) -> List[str]:
    tokens = tokenize(text)
    counts: Dict[str, int] = {}
    for token in tokens:
        if len(token) <= 2 or token in STOP_WORDS:
            continue
        counts[token] = counts.get(token, 0) + 1

    if not counts and tokens:
        return tokens[: min(len(tokens), limit)]

    sorted_tokens = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [token for token, _ in sorted_tokens[:limit]]


def shorten(text: str, limit: int = 90) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit].strip()}…"


def build_link_summary(
    note_embedding: List[float], note_tags: List[str], existing_notes: List[Dict[str, object]]
) -> str:
    if not existing_notes:
        return "Seed insight — establishes the first anchor in this workspace."

    scored = []
    for candidate in existing_notes:
        score = cosine_similarity(note_embedding, candidate["embedding"])
        if score > 0:
            scored.append((candidate, score))

    if not scored:
        return "Introduces a fresh concept that has not been captured yet."

    scored.sort(key=lambda entry: entry[1], reverse=True)
    closest, score = scored[0]
    shared_tags = [tag for tag in note_tags if tag in closest["tags"]]

    if score < 0.18:
        return "Adds a new direction — tags will help surface related notes as the corpus grows."
    if shared_tags:
        return f'Extends "{shorten(closest["content"])}" with more depth on {", ".join(shared_tags)}.'
    return f'Builds on "{shorten(closest["content"])}" with a complementary angle.'


def build_graphs(notes: List[Dict[str, object]]) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    if not notes:
        return {"noteGraph": {"nodes": [], "edges": []}, "tagGraph": {"nodes": [], "edges": []}}

    note_tags = {note["id"]: set(note["tags"]) for note in notes}
    tag_counts = Counter(tag for tags in note_tags.values() for tag in tags)
    tag_to_notes: Dict[str, Set[str]] = defaultdict(set)
    for note_id, tags in note_tags.items():
        for tag in tags:
            tag_to_notes[tag].add(note_id)

    def tag_weight(tag: str) -> float:
        count = tag_counts.get(tag, 1)
        return 1.0 / (1.0 + math.log(1 + count))

    note_nodes = [
        {
            "id": note["id"],
            "label": shorten(note["content"], 70),
            "createdAt": note["createdAt"],
            "tagCount": len(note_tags[note["id"]]),
        }
        for note in notes
    ]

    note_edges: List[Dict[str, object]] = []
    for idx, left in enumerate(notes):
        for right in notes[idx + 1 :]:
            shared = sorted(note_tags[left["id"]] & note_tags[right["id"]])
            if not shared:
                continue
            weight = sum(tag_weight(tag) for tag in shared)
            note_edges.append(
                {
                    "id": f'{left["id"]}-{right["id"]}',
                    "source": left["id"],
                    "target": right["id"],
                    "weight": weight,
                    "sharedTags": shared,
                }
            )

    note_edges.sort(key=lambda entry: entry["weight"], reverse=True)
    note_edges = note_edges[:60]

    tag_nodes = [{"id": tag, "label": tag, "noteCount": count} for tag, count in tag_counts.most_common()]

    tag_edges: List[Dict[str, object]] = []
    tags = list(tag_to_notes.keys())
    for idx, tag_a in enumerate(tags):
        for tag_b in tags[idx + 1 :]:
            shared_notes = sorted(tag_to_notes[tag_a] & tag_to_notes[tag_b])
            if not shared_notes:
                continue
            penalties = []
            summaries = []
            for note_id in shared_notes:
                tag_total = len(note_tags[note_id]) or 1
                penalties.append(1.0 / (1.0 + math.log(1 + tag_total)))
                source_note = next((note for note in notes if note["id"] == note_id), None)
                if source_note:
                    summaries.append(shorten(source_note["content"], 70))
            weight = sum(penalties)
            tag_edges.append(
                {
                    "id": f"{tag_a}-{tag_b}",
                    "source": tag_a,
                    "target": tag_b,
                    "weight": weight,
                    "sharedNotes": summaries[:5],
                }
            )

    tag_edges.sort(key=lambda entry: entry["weight"], reverse=True)
    tag_edges = tag_edges[:60]

    return {
        "noteGraph": {"nodes": note_nodes, "edges": note_edges},
        "tagGraph": {"nodes": tag_nodes, "edges": tag_edges},
    }


def collect_existing_tags(notes: List[Dict[str, object]]) -> List[str]:
    catalog = set()
    for note in notes:
        catalog.update(note.get("tags", []))
    return sorted(catalog)


def build_neighbor_context(
    note_embedding: List[float], existing_notes: List[Dict[str, object]], limit: int = 4
) -> List[Dict[str, str]]:
    scored = []
    for candidate in existing_notes:
        score = cosine_similarity(note_embedding, candidate["embedding"])
        if score <= 0:
            continue
        scored.append((candidate, score))
    scored.sort(key=lambda entry: entry[1], reverse=True)
    snippets = []
    for note, _ in scored[:limit]:
        snippets.append(
            {
                "summary": shorten(note["content"], 160),
                "tags": note["tags"],
            }
        )
    return snippets


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=4000, reload=True)
