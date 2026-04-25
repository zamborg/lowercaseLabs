# blackhole — architecture

## Ingest (voice → stored item)

```mermaid
sequenceDiagram
    actor User
    participant iOS as iOS App
    participant Engine as Dictation Engine<br/>(on-device)
    participant API as blackhole.fly.dev
    participant LLM as gpt-5.4-mini
    participant DB as SQLite (/data)

    User->>iOS: press & hold button
    iOS->>Engine: start session
    Engine-->>iOS: streaming partial transcripts
    iOS-->>User: live text updates

    User->>iOS: release button
    Engine-->>iOS: final transcript
    iOS-->>User: editable TextEditor

    User->>iOS: tap "Send to Blackhole"
    iOS->>API: POST /items {content}
    API->>LLM: classify transcript
    LLM-->>API: {type, title, tags, due_date}
    API->>DB: INSERT item
    API-->>iOS: Item
    iOS-->>User: toast (title + type)
```

---

## Auth (Sign in with Apple)

```mermaid
sequenceDiagram
    actor User
    participant iOS as iOS App
    participant Apple as Apple servers
    participant API as blackhole.fly.dev

    User->>iOS: tap Sign in with Apple
    iOS->>Apple: request identity token
    Apple-->>iOS: identity_token (JWT signed by Apple)
    iOS->>API: POST /auth/apple {identity_token}
    API->>Apple: fetch public keys
    Apple-->>API: JWKS
    API->>API: verify token signature + claims
    API-->>iOS: session_token (our JWT, 90 day expiry)
    iOS->>iOS: store in UserDefaults
    Note over iOS,API: all subsequent requests use Bearer session_token
```

---

## Feed (list items)

```mermaid
sequenceDiagram
    participant iOS as iOS App
    participant API as blackhole.fly.dev
    participant DB as SQLite

    iOS->>API: GET /items (Bearer token)
    API->>DB: SELECT * WHERE user_id ORDER BY created_at DESC
    DB-->>API: rows
    API-->>iOS: [Item]
    iOS->>iOS: replace local item cache
    iOS-->>iOS: render FeedView (all items, searchable)
    Note over iOS: swipe to delete → DELETE /items/{id}<br/>tap circle → PATCH /items/{id} {completed}
```

Feed, Epics, Notes, and Todos hydrate from the local item cache first, then refresh from `/items`. If the refresh fails, cached content remains visible.

---

## Feed search (natural language query)

```mermaid
sequenceDiagram
    actor User
    participant iOS as iOS App
    participant Engine as Apple Speech<br/>(on-device)
    participant API as blackhole.fly.dev
    participant LLM as gpt-5.4-mini
    participant DB as SQLite

    User->>iOS: use Feed search bar
    iOS->>Engine: dictate (optional)
    Engine-->>iOS: query text
    iOS->>API: POST /search {query}
    API->>DB: SELECT all user items
    DB-->>API: items
    API->>LLM: rank items by relevance to query
    LLM-->>API: {indices: [3, 0, 7, ...]}
    API-->>iOS: [Item] (sorted by relevance)
    iOS-->>User: results list
```

If `/search` is unavailable, Feed search falls back to simple local matching over cached title, content, tags, and item type.

---

## Component map

```
blackhole/
├── ios/                        iOS client (SwiftUI, iOS 17+)
│   ├── Sources/Dictation/      on-device transcription
│   │   ├── AppleSpeechDictationEngine   uses SFSpeechRecognizer
│   │   └── WhisperKitDictationEngine    uses Core ML (tiny.en model)
│   ├── Sources/Ingest/         hold-to-dictate + submit flow
│   ├── Sources/Feed/           feed search + notes list + shared item detail
│   ├── Sources/Search/         legacy search screen, not mounted in tabs
│   ├── Sources/Auth/           Sign in with Apple
│   └── Sources/API/            HTTP client → blackhole.fly.dev
│
└── backend/                    API server (FastAPI, Python 3.12)
    └── app/
        ├── main.py             routes: /auth/apple /items /search
        ├── auth.py             Apple JWT verification + session JWT
        ├── analysis.py         compatibility facade for LLM operations
        ├── agent/responses/    OpenAI Responses client, prompts, logs, tool registry
        └── db.py               SQLite via stdlib sqlite3, WAL mode
```

## Item classification

The model (`gpt-5.4-mini`) decides based on the raw transcript. No hard rules — it reads intent. Signals it uses in practice:

| Signal | Likely classification |
|---|---|
| Action verb ("call", "buy", "fix", "remind me") | todo |
| Explicit date/time ("by Friday", "tomorrow at 3") | todo + extracts due_date |
| Informational / reflective content | note |
| Ideas, observations, references | note |
| Project/workstream/goal language ("create an epic", "Q2 launch") | epic |

Document-like captures should generally stay as one note because the raw user content is the source of truth. The model only splits when the user clearly gives distinct todos or asks for separate notes/items.

To tune this, edit the prompts in `backend/app/agent/responses/prompts.py`.
