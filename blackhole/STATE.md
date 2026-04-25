# blackhole — state

Last updated: 2026-04-24

Read this first if you need to orient quickly. This is the current operational state of the repo, not a product pitch.

## What this is

`blackhole` is a voice-first iOS app plus a small FastAPI backend.

- iOS captures speech, manages auth, and renders the app UI.
- The backend verifies Apple sign-in, stores items in SQLite, calls OpenAI to structure transcripts, and serves a simple admin dashboard.

The core object is `Item`:

- `id`
- `content`
- `title`
- `type` = `note`, `todo`, or `epic`
- `due_date` nullable ISO string
- `completed` bool
- `tags` list of strings
- `created_at`
- `updated_at`

Canonical model definitions:

- iOS: [ios/Sources/Models/Item.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Models/Item.swift)
- backend: [backend/app/schemas.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/schemas.py)

## Product shape today

The signed-in app has 5 tabs:

- `Void`: capture voice, edit transcript, send to backend
- `Feed`: all items by default, with text/voice search at the top
- `Epics`: epics only
- `Notes`: notes only
- `Todos`: open vs done todos, sorted with due dates first

Top-level app entry:

- [ios/App/blackholeApp.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/App/blackholeApp.swift)
- [ios/App/RootView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/App/RootView.swift)

## Backend shape

Primary backend file:

- [backend/app/main.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/main.py)

Primary routes:

- `POST /auth/apple`
- `POST /items`
- `GET /items`
- `PATCH /items/{item_id}`
- `DELETE /items/{item_id}`
- `POST /search`
- `GET /admin`
- `POST /admin/login`
- `POST /admin/logout`
- `GET /admin/api/overview`

Supporting modules:

- [backend/app/auth.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/auth.py): Apple token verification, session JWTs
- [backend/app/analysis.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/analysis.py): compatibility facade for transcript analysis and search ranking
- [backend/app/agent/responses/](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/agent/responses): OpenAI Responses client, prompt configs, LLM log shaping, and tool registry
- [backend/app/db.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/db.py): SQLite schema and queries

Storage notes:

- SQLite only, no ORM, no migration system
- default DB path is `/data/blackhole.db`
- Fly volume is expected for persistence in production
- schema is created in `db.init_db()`
- iOS also keeps a per-user JSON cache of items in Application Support for offline/cold-launch reads
- successful item list/create/edit/delete calls update the local iOS cache

If you change stored fields, update:

- `db.init_db()`
- route serialization in `main.py`
- iOS `Item` decoding and the item cache in `ios/Sources/API/APIClient.swift`
- Pydantic schemas
- iOS `Item`
- backend tests

## iOS shape

### Auth

- Sign in with Apple on device
- Apple identity token is exchanged with backend for a long-lived session token
- session token is stored in `UserDefaults`

Files:

- [ios/Sources/Auth/AuthViewModel.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Auth/AuthViewModel.swift)
- [ios/Sources/Auth/SignInView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Auth/SignInView.swift)

### API client

Single shared actor:

- [ios/Sources/API/APIClient.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/API/APIClient.swift)

Notes:

- base URL is hardcoded to `https://blackhole.fly.dev`
- create item returns `[Item]`, not `Item`, because one transcript may be split into multiple items

### Dictation

Dictation is abstracted behind `DictationEngine` / `DictationSession`.

Files:

- [ios/Sources/Dictation/DictationEngine.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Dictation/DictationEngine.swift)
- [ios/Sources/Dictation/DictationEngineCoordinator.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Dictation/DictationEngineCoordinator.swift)
- [ios/Sources/Dictation/AudioCapturePipeline.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Dictation/AudioCapturePipeline.swift)

Engines:

- Apple Speech: [ios/Sources/Dictation/AppleSpeechDictationEngine.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Dictation/AppleSpeechDictationEngine.swift)
- WhisperKit: [ios/Sources/Dictation/WhisperKitDictationEngine.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Dictation/WhisperKitDictationEngine.swift)

Important current behavior:

- app default engine is `WhisperKit`
- WhisperKit model warmup is kicked off at app startup
- WhisperKit uses a shared runtime
- downloaded model files are cached under Application Support
- later launches should prefer local-only load and avoid re-downloading unless the cache is missing/bad
- process-cold startup can still be slow because local load still includes model load + runtime init + prewarm

Important nuance:

- avoiding re-download is solved
- “cold process -> model ready” is not free and can still be noticeable

### Feature screens

- Ingest: [ios/Sources/Ingest/IngestViewModel.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Ingest/IngestViewModel.swift), [ios/Sources/Ingest/IngestView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Ingest/IngestView.swift)
- Feed and Notes: [ios/Sources/Feed/FeedView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Feed/FeedView.swift)
- Todos: [ios/Sources/Todos/TodosView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Todos/TodosView.swift)
- Edit: [ios/Sources/Edit/EditItemView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Edit/EditItemView.swift)
- Settings: [ios/Sources/Settings/SettingsView.swift](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ios/Sources/Settings/SettingsView.swift)

Search nuance:

- search ranking is server-side via `POST /search`
- feed search voice capture currently uses Apple Speech directly, not the user-selected dictation engine

## OpenAI usage

Current analysis model in code:

- `gpt-5.4-mini-2026-03-17`

Used for:

- transcript -> structured item extraction
- search ranking over existing items

Implementation notes:

- OpenAI calls use the Responses API through `backend/app/agent/responses/client.py`
- prompts and JSON schemas live in `backend/app/agent/responses/prompts.py`
- LLM log payload shaping lives in `backend/app/agent/responses/logging.py`
- future tool definitions should be registered through `backend/app/agent/responses/tools.py`

Transcript analysis behavior:

- backend sends recent items as context when classifying a new transcript
- one transcript may return multiple structured items
- fallback if analysis fails: create a single plain note from the raw transcript
- search LLM calls are logged and fall back to substring matching on failure

Files:

- [backend/app/analysis.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/analysis.py)
- [backend/app/agent/responses/prompts.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/app/agent/responses/prompts.py)

## Admin dashboard

The admin UI is server-rendered HTML assembled inside `backend/app/main.py`.

It is intentionally simple and currently includes:

- aggregate item/user metrics
- due/overdue views
- recent items
- LLM logs

Credentials come from env:

- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`

This is not a separate frontend app. If you touch admin UI, you are editing Python string-built HTML/CSS/JS in `main.py`.

## Tests

Current automated coverage is backend-only.

Run:

```bash
cd backend
python -m pytest tests/ -v
```

Live OpenAI smoke:

```bash
BLACKHOLE_RUN_OPENAI_TESTS=1 OPENAI_API_KEY=<key> python -m pytest tests/ -v -k smoke
```

Test file:

- [backend/tests/test_backend_integration.py](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/backend/tests/test_backend_integration.py)

What is actually covered:

- `/items` stores LLM logs and exposes them in admin
- OpenAI client calls use `max_completion_tokens`
- prior-item context injection behavior

What is not meaningfully covered:

- iOS UI behavior
- iOS dictation flows
- end-to-end Apple sign-in
- real WhisperKit startup behavior

## Current hotspots / likely next edits

- Whisper settings still do not expose `clear cache` / `re-download model`
- Whisper startup is better than before but still bounded by local model load + prewarm
- backend still uses deprecated FastAPI `@app.on_event("startup")`
- backend still uses `datetime.utcnow()`
- `main.py` is doing a lot: API routes, admin auth, admin rendering, admin metrics

## If you are changing X

If changing transcript analysis:

- edit prompts/model usage in `backend/app/agent/responses/prompts.py`
- confirm `/items` still handles multi-item output correctly
- run backend tests

If changing item shape:

- update both model definitions first
- then backend DB/schema/serialization
- then iOS decoding/editing/rendering

If changing dictation behavior:

- inspect `DictationEngine`, `DictationEngineCoordinator`, and `AudioCapturePipeline`
- remember ingest and feed search do not use identical engine selection behavior

If changing auth:

- inspect both Apple identity-token verification on backend and `UserDefaults` token persistence on iOS

## Related docs

- [ARCHITECTURE.md](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/ARCHITECTURE.md): broader flow overview
- [AGENTS.md](/Users/zubinaysola/Documents/personal/lowercaseLabs/blackhole/AGENTS.md): test / deploy / commit commands
