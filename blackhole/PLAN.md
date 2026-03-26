# blackhole v1 Plan

## Summary
Build `blackhole` as a new capture-first iOS app that reuses the existing FastAPI/Fly backend codebase but ships under a separate `blackhole` product surface. V1 is intentionally simple: remote-authoritative data, remote transcription by default, notes stored as markdown in the database, todos stored as structured records, and note recall powered by backend lexical search only. The implementation must be stream-capable from day one, but true live partial transcript text is deferred; v1 will use an event-stream transcription session API that can later emit partials without changing client interfaces.

## Chosen defaults
- `blackhole` is a new standalone iOS app target in the existing Xcode workspace; current `theVoid` features remain untouched.
- The existing FastAPI backend and Fly deployment are reused; all new routes are namespaced under `/blackhole/*`.
- V1 authority model is `remote-only`.
- V1 transcription default is `remote`.
- V1 search is `FTS-first`.
- Notes are stored as database rows with markdown bodies.
- Capture routing is `auto-route with confirmation only when confidence is low`.
- Streaming in v1 means `stream-capable API`, not true live partial transcript text during recording.
- Search scope in v1 is `notes only`.
- Chat over notes, vector search, query expansion, reranking, and on-device retrieval are out of scope for v1.

## Goals
- Preserve the one-button capture UX that already works well in `theVoid`.
- Support three top-level flows: create note, create todo, search notes.
- Make transcription, routing, and retrieval modular so providers can be swapped later.
- Keep the design compatible with a future qmd-like hybrid retrieval stack.
- Avoid entangling capture, inference, persistence, and retrieval in one monolithic app model.

## Out of scope for v1
- True realtime backend audio streaming with partial transcript text while recording.
- Local-first sync or offline-first creation.
- Vector embeddings, chunk search, RRF fusion, query expansion, reranking.
- Chat UI over notes.
- Markdown file storage as the canonical note format.
- Social features from `theVoid`.

## Product surface
- Tab 1: `Capture`
  - Single central record button.
  - Post-stop remote transcription.
  - Shows recording state, upload/transcribing state, final transcript preview, and routing outcome.
  - Auto-creates note or todo when confidence is high.
  - Shows confirmation sheet when confidence is below threshold.
  - If the route is `search`, opens note search results and does not persist a capture record.
- Tab 2: `Todos`
  - Default view is open todos.
  - Top filter control: `Open`, `All`, `Done`.
  - Tap into a todo detail screen for edit/update.
  - Bottom or top compact voice-add entry point that always creates a todo capture flow.
- Tab 3: `Notes`
  - Reverse-chronological note timeline.
  - Text search bar plus voice-search button.
  - Tap note to open detail/edit view.

## Backend architecture
- Reuse the current FastAPI app and auth stack.
- Add a `blackhole` module group with separate models, schemas, routers, and services.
- Reuse current Apple auth/session handling; `blackhole` records are keyed by the same `users.id`.
- Reuse existing object storage for temporary audio uploads.
- Reuse existing OpenAI client wiring for v1 remote transcription and route classification.
- Keep `theVoid` API paths and tables unchanged.

## iOS architecture
- Create separate `blackhole` feature modules instead of extending the current `AppModel` pattern.
- Required Swift service boundaries:
  - `AudioCaptureService`
  - `TranscriptionSessionClient`
  - `CaptureRouterClient`
  - `NotesRepository`
  - `TodosRepository`
  - `SearchRepository`
  - `AuthSessionStore`
- Required stream-native client interface:
  - `TranscriptionSessionClient.start(audioURL:mode:) async throws -> AsyncThrowingStream<TranscriptEvent, Error>`
- `TranscriptEvent` must support future partials even if v1 does not emit them:
  - `sessionCreated`
  - `uploading`
  - `transcribing`
  - `partialTranscript`
  - `finalResult`
  - `failed`

## Data model
- `blackhole_notes`
  - `id`
  - `user_id`
  - `title`
  - `body_markdown`
  - `tags` JSON array
  - `source_capture_id` nullable
  - `created_at`
  - `updated_at`
  - `archived_at` nullable
- `blackhole_todos`
  - `id`
  - `user_id`
  - `title`
  - `description_markdown`
  - `priority` enum: `low | normal | high`
  - `status` enum: `open | completed | canceled`
  - `deadline_at` nullable
  - `tags` JSON array
  - `source_capture_id` nullable
  - `created_at`
  - `updated_at`
  - `completed_at` nullable
- `blackhole_captures`
  - `id`
  - `user_id`
  - `mode` enum: `note | todo`
  - `audio_object_key`
  - `transcript_text`
  - `transcript_provider`
  - `routing_target` enum: `note | todo`
  - `routing_confidence`
  - `routing_payload` JSON
  - `created_record_id` nullable
  - `created_at`
  - `updated_at`
- `blackhole_transcription_sessions`
  - `id`
  - `user_id`
  - `mode` enum: `auto | note | todo | search`
  - `status` enum: `created | uploading | transcribing | completed | failed`
  - `audio_object_key`
  - `final_transcript` nullable
  - `route` nullable
  - `route_confidence` nullable
  - `route_payload` JSON nullable
  - `error_message` nullable
  - `created_at`
  - `updated_at`

## Routing rules
- Modes:
  - `auto`: classify into `note`, `todo`, or `search`.
  - `note`: skip route classification and create note draft.
  - `todo`: skip route classification and create todo draft.
  - `search`: skip persistence and produce search query text only.
- Auto-route confidence threshold: `0.80`.
- If route confidence is below `0.80`, client shows confirmation sheet with three actions: `Create Note`, `Create Todo`, `Search Notes`.
- If route classification fails, default fallback is `note`.
- Todo routing should bias toward imperative/commitment language; all ambiguous reflective speech falls back to note.

## Note and todo draft payloads
- Note draft payload from router:
  - `title`
  - `body_markdown`
  - `tags`
- Todo draft payload from router:
  - `title`
  - `description_markdown`
  - `priority`
  - `deadline_at`
  - `tags`
- Search payload from router:
  - `query_text`

## Public backend APIs
- `POST /blackhole/transcriptions`
  - multipart upload of recorded audio
  - body includes `mode`
  - returns `transcription_session_id`
- `GET /blackhole/transcriptions/{id}/events`
  - SSE endpoint
  - emits `uploading`, `transcribing`, `partial_transcript`, `final_result`, `error`
- `POST /blackhole/notes`
  - create note from confirmed or auto-routed draft
- `GET /blackhole/notes`
  - cursor-paginated, newest first
- `GET /blackhole/notes/{id}`
- `PATCH /blackhole/notes/{id}`
- `POST /blackhole/todos`
  - create todo from confirmed or auto-routed draft
- `GET /blackhole/todos`
  - filter by `status`
- `GET /blackhole/todos/{id}`
- `PATCH /blackhole/todos/{id}`
- `GET /blackhole/search`
  - params: `q`, `limit`, `cursor`
  - returns note results only in v1

## Search implementation
- Production search is PostgreSQL lexical search over notes using weighted fields:
  - title weight highest
  - tags weight medium
  - body weight standard
- Ranking order:
  - lexical relevance score descending
  - `updated_at` descending as tiebreaker
- Dev/test fallback:
  - SQLite-compatible normalized text search using `LIKE`/portable SQL so local development remains functional.
- Search returns:
  - note id
  - title
  - snippet
  - tags
  - score
  - updated_at
- Search ignores archived notes.

## Transcription implementation
- V1 uses current backend OpenAI client infrastructure as the remote transcription provider.
- The backend transcription session API must be event-driven even if the provider returns only a final transcript.
- V1 SSE behavior:
  - emit `uploading`
  - emit `transcribing`
  - emit `final_result` with transcript and route payload
  - emit `error` on failure
- True live partial transcript emission is deferred, but the event schema must already support it.

## Audio retention
- Audio is temporary in v1.
- Successful note/todo creation deletes the audio object asynchronously within 24 hours.
- Failed transcription retains audio for retry/debug for up to 24 hours, then cleanup removes it.
- Search-mode voice queries do not create persistent capture records and do not retain audio after session completion.

## Required config additions
- `blackhole_router_model` default `gpt-4.1-mini`
- `blackhole_auto_route_threshold` default `0.80`
- `blackhole_audio_retention_hours` default `24`
- `blackhole_search_default_limit` default `20`
- `blackhole_search_max_limit` default `50`

## Migration and compat plan
1. Add new Alembic migrations for all `blackhole_*` tables and indexes.
2. Add backend routers under `/blackhole/*`.
3. Add iOS `blackhole` target and shared client/auth dependencies.
4. Do not rename or mutate `theVoid` tables, routes, or UI flows.
5. Keep the current backend auth/session model compatible for both apps.

## Implementation phases
1. Foundation
   - Carve current backend OpenAI transcription and classification code into reusable provider services.
   - Define shared schemas for transcription sessions, route payloads, notes, todos, and search results.
   - Create new DB tables and migrations.
2. Backend core
   - Implement transcription session creation and SSE events.
   - Implement note CRUD, todo CRUD, and note search endpoints.
   - Add cleanup job for temporary audio.
3. iOS shell
   - Create `blackhole` app target and tab shell.
   - Build `Capture`, `Todos`, and `Notes` screens.
   - Implement stream-native transcription client.
4. Routing UX
   - Add final transcript review and low-confidence confirmation sheet.
   - Wire auto-create path for high-confidence note/todo routes.
   - Wire search route into Notes search results.
5. Hardening
   - Add metrics, error states, retries, and pagination.
   - Add edit/update flows for note and todo detail screens.

## Tests and scenarios
- Backend unit tests
  - route classifier returns valid schema for note, todo, and search
  - low-confidence route does not auto-commit
  - route failure defaults to note
  - note CRUD enforces user ownership
  - todo CRUD enforces user ownership
  - archived notes are excluded from search
  - SQLite fallback search works in local tests
- Backend integration tests
  - create transcription session, consume SSE, receive final result
  - auto-routed note creation from final route payload
  - auto-routed todo creation from final route payload
  - search-mode query returns results without persisting a note/todo
  - audio cleanup job removes expired temporary audio
- iOS tests
  - capture screen state machine: idle, recording, uploading, transcribing, result, failed
  - high-confidence note path auto-creates note
  - high-confidence todo path auto-creates todo
  - low-confidence result shows confirmation sheet
  - voice search path opens search results and persists nothing
  - todos tab filtering works
  - note timeline pagination works
- End-to-end acceptance
  - user records “remember to file taxes tomorrow” and sees a todo created
  - user records “thinking about product direction today” and sees a note created
  - user records “find my note about onboarding ideas” and lands in note search results
  - user can edit created note/todo from detail view

## Monitoring and rollout
- Add backend metrics:
  - transcription session count
  - transcription latency
  - route distribution by `note | todo | search`
  - low-confidence rate
  - search latency
  - zero-result search rate
  - note creation count
  - todo creation count
- Add structured logs for transcription session lifecycle and route classification.
- Release `blackhole` behind a separate app target and endpoint namespace.
- Keep rollout internal/TestFlight first; do not migrate existing `theVoid` users automatically.

## Future extension path to qmd-like retrieval
- Add note chunk table with chunk metadata and offsets.
- Add embedding provider abstraction.
- Add vector search backend.
- Add query expansion and reranker interfaces.
- Add fusion strategy interface with RRF weighting.
- Keep search result contract stable so Notes UI does not change when hybrid retrieval is added.

## Assumptions
- Apple sign-in remains the auth mechanism for v1.
- `blackhole` uses the existing backend deployment and database stack.
- Search quality in v1 is expected to be “good enough lexical recall,” not semantic recall.
- Users accept remote transcription in v1.
- The implementation prioritizes modular seams over maximum v1 feature count.
