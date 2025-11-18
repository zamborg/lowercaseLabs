# CNTX — Context Without the O

CNTX is a lightweight “research sparks” notebook. Researchers drop raw notes, CNTX auto-tags + links them, and teammates can search or skim a conceptual map without waiting for the next sync.

## Features

- **Slack-speed ingest**: zero-ceremony composer captures the raw note verbatim.
- **Automatic enrichment**: heuristic tags + relational summaries attach under each note (feel free to plug in your own AI pipeline later).
- **Semantic retrieval**: embedding-based search handles both exact recollection (“that tool calling bug from last Tuesday…”) and fuzzy topic queries.
- **Manual tagging + editing**: assign your own tags, tweak content later, and see those updates ripple through retrieval + graphs.
- **Concept map**: dual graphs show note↔note and tag↔tag associations, weighted by tag uniqueness.
- **Optional AI enrichment**: toggle “Use AI for tags + description” to let GPT-5-mini generate tags + note blurbs (manual tags still honored).

## Stack

- **Backend**: Python 3 + FastAPI + SQLite (embedding + metadata generation currently heuristic, swap in GPT when ready).
- **Frontend**: React + Vite + TypeScript.

## Quick Start

### 1. API (FastAPI)

```bash
cd /Users/zubinaysola/Documents/personal/lowercaseLabs/cntx
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
# optional: export OPENAI_API_KEY=sk-... to enable AI tagging
uvicorn backend.main:app --reload --port 4000
```

Notes are stored in `data/cntx.db`. Stop the server with `Ctrl+C`. Keep the venv active while you’re developing backend features.

### 2. Frontend (Vite dev server)

```bash
cd /Users/zubinaysola/Documents/personal/lowercaseLabs/cntx/frontend
npm install          # run once
npm run dev          # visit the printed URL (default http://localhost:5173)
```

The frontend talks to `http://localhost:4000` by default. To point elsewhere set `VITE_API_BASE_URL` before building/running: `VITE_API_BASE_URL=https://your-api npm run dev`.

### 3. Production build check

```bash
# With the backend still running
cd frontend
npm run build
```

This runs TypeScript + Vite builds to ensure the UI compiles cleanly.

## Data & Extensibility Notes

- Notes live in `data/cntx.db`.
- Embeddings use a hashed bag-of-words vector + cosine similarity for now.
- Link summaries explain how each new note relates to the closest existing entry. When the AI toggle is on, GPT-5-mini provides tags + descriptions via `backend/ai.py`.
- If you want to swap in a different model or prompt, edit `backend/ai.py` (the rest of the pipeline is already stubbed for structured outputs).

## Roadmap Seeds

1. Replace the heuristic metadata functions with GPT-5-mini calls (wrap them in a new module to keep `main.py` slim).
2. Add Slack or CLI ingest endpoints for true “dump while coding” workflows.
3. Add filters + timeline views to the visualization panel.
