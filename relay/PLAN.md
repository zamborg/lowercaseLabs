# relay — build plan

> Working name: **relay** (placeholder; rename freely). This document is the full
> execution plan for an agent building the app. Read it end to end before
> starting. Execute phases in order; **Phase 0 is a gate** — stop and report
> its results before building on top of it.

---

## 1. What we're building

A voice-first "better Siri" for iPhone. You trigger it (Action Button, Back Tap,
Control Center, Siri phrase), say something, and it does the thing:

- "Text Sam I'm running ten minutes late"
- "Skip this song"
- "Get me an Uber home"
- "Remind me to call the dentist tomorrow at 9"
- "What's 18% of 64?" (just answers)

How: a single **router shortcut** dictates your speech, hands the transcript to
our app's **App Intent**, which asks an LLM (user's own OpenAI or Anthropic API
key) to choose one **module** from a known registry and fill in its arguments.
The router then runs that module.

### Principles

- **Local-only.** No backend, no analytics, no accounts. The only network calls
  are to the LLM provider the user configured.
- **Bring your own key.** User pastes an OpenAI and/or Anthropic API key; stored
  in the iOS Keychain. Never logged, never committed.
- **We own the registry.** The app ships knowing its modules (id, description,
  input schema, shortcut name). We do not try to read the user's shortcut
  library.
- **Fast by default, confirm when it matters.** Harmless actions fire
  immediately; anything that sends, buys, or calls shows a confirmation.
- **Clean slate.** Do not copy code, patterns, names, or structure from any
  other project in this repository. Everything lives under `relay/`.

### Out of scope for v1

App Store onboarding polish, LLM-generated shortcuts, multi-turn conversation,
on-device models, iPad/Mac/Watch targets.

---

## 2. Architecture

```
 Trigger (Action Button / Back Tap / Control / "Hey Siri, Relay")
    │
    ▼
 ┌───────────────── Router shortcut ("Relay") ─────────────────┐
 │ 1. Dictate Text                                             │
 │ 2. Relay: Decide (our App Intent, input = transcript)       │──► LLM API
 │      returns JSON text                                      │◄── (tool call)
 │ 3. Get Dictionary from Input                                │
 │ 4. If action == "run":                                      │
 │      If confirm: Show Alert(summary) [Cancel stops]         │
 │      Run Shortcut <shortcut name> with input <input dict>   │
 │    Otherwise if action == "respond" / "clarify":            │
 │      Speak Text / Show Result <speak>                       │
 └─────────────────────────────────────────────────────────────┘
    │
    ▼
 Module shortcut, e.g. "Relay · Send Message"
   Get Dictionary from Input → Get values → Send Message (system action)
```

Why this shape: a third-party app cannot silently run a user's shortcuts (the
only API is the `shortcuts://` URL, which foregrounds the Shortcuts app). By
making the *shortcut* the entry point and our app a background "brain" action,
everything stays inside Shortcuts with no app switching. System capabilities we
can't call from code (Send Message, Skip Forward, Start Timer, Set Focus…) live
in small module shortcuts; capabilities we *can* do in code live as our own App
Intents.

### Two kinds of modules

| Kind | Executed by | Examples | Adding one requires |
|---|---|---|---|
| **Shortcut module** | A module shortcut run by the router | send message, skip track, start timer, set focus, flashlight | registry entry + a new shortcut on the phone |
| **Native module** | Our app, inside the Decide intent (or a second intent) | open URL / deep link (Uber, Maps), add reminder (EventKit), answer a question | registry entry + Swift code + app rebuild |

Long-term, prefer native modules: they're testable and tunable in Swift, and
the router never changes.

---

## 3. Tech choices

- **Swift 6 / SwiftUI**, iOS deployment target **17.0** (user's device is an
  iPhone 14; confirm its installed iOS version and test there).
- **App Intents** framework for the Decide intent and native modules. Intents
  live in the **main app target** (not an extension) for v1, so they share the
  app's Keychain and storage with no App Group setup. Set
  `static let openAppWhenRun = false`.
- **XcodeGen** (`project.yml`) to generate the Xcode project. Do not hand-edit
  `.pbxproj`. Commit `project.yml`; ignore the generated `.xcodeproj` (or commit
  it, but always regenerate rather than edit).
- **Swift Package `RelayCore`** for all logic that doesn't need UIKit: registry,
  prompt building, LLM clients, response parsing, the decision contract. It
  builds on macOS too, so it can be unit-tested and evaluated from the command
  line without a phone.
- **Networking:** plain `URLSession` + `Codable`. **No third-party SDKs.**
- **Persistence:** SwiftData for history; Keychain (Security framework, small
  wrapper) for API keys; `UserDefaults` for non-secret settings.
- **Speech:** Shortcuts' built-in **Dictate Text** action (on-device, free). We
  do not record audio ourselves in v1.

---

## 4. Repository layout

```
relay/
  PLAN.md                  ← this file
  README.md                ← how to build, install, set up the router
  project.yml              ← XcodeGen spec
  App/
    RelayApp.swift
    Views/                 ← Settings, Modules, History, Onboarding
    Intents/
      DecideIntent.swift
      AppShortcuts.swift   ← AppShortcutsProvider (Siri phrases)
    Native/                ← native module executors (Reminders, OpenURL…)
    Storage/               ← Keychain wrapper, SwiftData models
  Packages/RelayCore/
    Package.swift
    Sources/RelayCore/
      Registry/            ← Module, InputSchema, builtInModules
      Decision/            ← Decision contract (encode/decode JSON)
      Providers/           ← LLMProvider protocol, OpenAIProvider, AnthropicProvider
      Router/              ← Router: transcript + context → Decision
      Prompt/              ← system prompt builder
    Sources/relay-eval/    ← macOS CLI: runs evals/utterances.jsonl against a real API
    Tests/RelayCoreTests/
  evals/
    utterances.jsonl       ← {"utterance": ..., "expect": {"module": ..., "input": {...}}}
  shortcuts/
    specs/                 ← one markdown spec per shortcut (exact action list)
    src/                   ← XML plist sources (after export, see §9)
    signed/                ← gitignored build output
    Makefile               ← `make sign` (macOS only)
```

Add to `.gitignore` (inside `relay/` or repo root): `relay/shortcuts/signed/`,
`*.xcodeproj` if generated, any `.env`/key files.

---

## 5. The decision contract (Decide intent → router)

The Decide intent returns a **JSON string**. The router parses it with
"Get Dictionary from Input". Keep it flat and stable — the router depends on it.

```json
{ "action": "run",
  "shortcut": "Relay · Send Message",
  "input": { "recipient": "+15551234567", "body": "Running ten minutes late" },
  "confirm": true,
  "summary": "Text Sam: \"Running ten minutes late\"" }

{ "action": "done",   "speak": "Reminder set for tomorrow at 9." }   // native module already executed
{ "action": "respond","speak": "18% of 64 is 11.52." }
{ "action": "clarify","speak": "Who should I text?" }
{ "action": "error",  "speak": "I couldn't reach OpenAI. Check your key in Relay." }
```

Rules:
- `run` only for shortcut modules. Native modules execute inside the intent and
  return `done`.
- `confirm` comes from the module's registry entry (`requiresConfirmation`), not
  from the LLM. The LLM never gets to skip confirmation.
- Every response has either `shortcut` or `speak`; the router never has to guess.
- `clarify` ends the run in v1 (just speaks). Multi-turn is a later phase.

---

## 6. Registry

```swift
struct Module: Sendable, Codable {
  let id: String                 // "send_message" — also the LLM tool name
  let description: String        // written for the LLM; say when to use AND when not to
  let input: InputSchema         // JSON Schema subset: object of string/number/bool/enum fields, required list
  let kind: Kind                 // .shortcut(name: String) | .native
  let requiresConfirmation: Bool
}
```

### v1 modules

| id | kind | input | confirm | notes |
|---|---|---|---|---|
| `send_message` | shortcut "Relay · Send Message" | recipient, body | yes | Resolve recipient name → phone number in-app via Contacts framework before returning (more reliable than Find Contacts in Shortcuts). If 0 or >1 matches, return `clarify`. |
| `call_contact` | native (`tel:` URL) | recipient | yes | Same contact resolution. |
| `media_control` | shortcut "Relay · Media" | command: play/pause/next/previous | no | Uses system Play/Pause and Skip actions; works with whatever is playing (Spotify included). |
| `start_timer` | shortcut "Relay · Timer" | minutes, seconds | no | |
| `set_alarm` | shortcut "Relay · Alarm" | time (HH:mm), label | no | |
| `add_reminder` | native (EventKit) | title, due (ISO 8601, optional) | no | LLM gets current date/time + time zone in the prompt. |
| `create_note` | shortcut "Relay · Note" | text | no | |
| `directions` | native (open Apple Maps URL) | destination, mode: drive/walk/transit | no | "home" resolves to the saved home address. |
| `request_ride` | native (open Uber deep link) | destination | yes | Opens Uber with destination prefilled; user confirms in Uber. Verify the deep link format against Uber's current docs. |
| `set_focus` | shortcut "Relay · Focus" | focus: dnd/sleep/work/off | no | |
| `flashlight` | shortcut "Relay · Flashlight" | on: bool | no | |
| `answer` | built-in | text | no | General questions, math, quick facts → `respond`. |
| `clarify` | built-in | question | no | When the request is ambiguous or missing required info. |

Settings provide: home address, optional nickname map ("mom" → contact).

---

## 7. LLM routing

- **One tool per module**; tool name = `module.id`, parameters = `module.input`.
  Force a tool call (OpenAI `tool_choice: "required"`, Anthropic
  `tool_choice: {"type": "any"}`). Exactly one call is used; ignore extras.
- **System prompt** (built in `RelayCore/Prompt`), keep short:
  - You are the router for a voice assistant on the user's iPhone.
  - Current date, time, time zone, locale.
  - Pick exactly one tool. Prefer `clarify` over guessing a recipient or a destination.
  - The transcript comes from dictation and may contain recognition errors; infer intent.
  - For `answer`, reply in one or two spoken sentences.
- **Providers:** `protocol LLMProvider { func route(system:, transcript:, tools:) async throws -> ToolCall }`.
  - `OpenAIProvider` — use the Chat Completions or Responses API with tools;
    pick one and keep it consistent.
  - `AnthropicProvider` — Messages API with tools (`x-api-key`,
    `anthropic-version` headers).
- **Models:** never hard-code a guessed model name. The model is a user
  setting. In Settings, populate the picker by calling each provider's list-models
  endpoint with the user's key. For testing, the user intends to use a current
  small/fast OpenAI model; for Anthropic, default to `claude-haiku-5-5`.
- **Latency budget:** aim for under ~2 s from end of dictation to action. Use a
  small model, low `max_tokens`, a 10 s request timeout, and no retries in the
  intent path (App Intents have a limited background run time).
- **Validation:** decode tool arguments against the module schema. On failure,
  return `error` with a short spoken message; log the raw response to history.

---

## 8. App UI (minimal, functional)

Tabs or a single list-based screen:

1. **Setup** — provider picker, API key fields (secure, saved to Keychain,
   "Test key" button calls list-models), model picker, home address, router
   setup checklist (link to the router shortcut, how to bind it to Action Button
   or Back Tap).
2. **Modules** — the registry: each module's description, inputs, whether it
   needs confirmation, and for shortcut modules whether the user has installed
   it (best effort: just a checklist the user ticks).
3. **History** — each run: transcript, provider/model, chosen module + args,
   decision JSON, latency, error if any. Tap to copy. This is the main tuning
   tool. Allow "clear history".
4. **Try it** — a text field that runs the same Router as the intent (without
   executing shortcut modules), so routing can be tested in-app without dictation.

---

## 9. Shortcuts: authoring and installing

The router and module shortcuts are built **by hand once**, then kept as
diffable sources.

1. For each shortcut, write an exact spec in `shortcuts/specs/<name>.md`
   (ordered action list, every parameter, every variable).
2. The user builds it in the Shortcuts app from the spec.
3. On a Mac: export the shortcut file, convert to XML
   (`plutil -convert xml1 <file>`) and commit it to `shortcuts/src/`. This also
   reveals the real action identifiers for writing future ones.
4. To push an edited or new shortcut: `make sign` runs
   `shortcuts sign --mode anyone --input src/X.shortcut --output signed/X.shortcut`
   (macOS only, needs to be signed into iCloud). Open the signed file on the Mac
   to import; it syncs to the iPhone via iCloud. AirDrop also works.

Module shortcut convention: input is a dictionary. First action is "Get
Dictionary from Input", then "Get Dictionary Value" per field. Module names
exactly match the registry's `shortcut` names.

---

## 10. Phases

Commit at the end of each phase with a clear message. Do not move on until
the acceptance criteria are met.

### Phase 0 — Spike (gate)

Goal: prove the three assumptions the design depends on. Do not build anything
else first.

- Minimal app target with one `DecideIntent` (`openAppWhenRun = false`) that
  returns a **hard-coded** JSON string, and one that reads a value from the
  Keychain and makes a real HTTPS request.
- A router shortcut: Dictate Text → Decide → Get Dictionary → Run Shortcut
  using the **`shortcut` value from the dictionary as the shortcut to run** →
  a test module that shows an alert with its input.

Acceptance (report each as pass/fail to the user):
1. **Run Shortcut by variable name works.** If it does not, the fallback is a
   router built from an If / Otherwise If chain matching `shortcut` against each
   known module name. Report which approach is in use.
2. Decide runs from the Shortcut **without opening the app**, can read the
   Keychain, and completes a network call.
3. End-to-end latency from end of dictation to the test module, measured on the
   device.
4. Works when triggered from the Lock Screen / Action Button, or note what
   requires unlock.

**Stop here and report to the user before continuing.**

### Phase 1 — RelayCore + evals

- `Module`, `InputSchema`, `Decision`, built-in v1 registry (§6).
- Prompt builder, `OpenAIProvider`, `AnthropicProvider`, `Router`.
- Unit tests with a mocked provider: schema → tool JSON, tool call → Decision,
  confirmation comes from the registry, invalid args → `error`.
- `relay-eval` CLI: reads `evals/utterances.jsonl` (write ~40 utterances
  covering every module, ambiguous cases that should `clarify`, and
  dictation-style typos), reads the key from an environment variable, prints
  accuracy per module and latency. Never print the key.

Acceptance: `swift test` passes; `relay-eval` runs against a real key and
reports results.

### Phase 2 — App shell

- XcodeGen project, Keychain wrapper, settings, Setup / Modules / History /
  Try it screens (§8), SwiftData history model.

Acceptance: builds and runs on the device; keys persist; "Test key" works;
"Try it" shows the routed decision and records history.

### Phase 3 — Decide intent, for real

- `DecideIntent` calls `Router`, runs native modules (Reminders, Maps, Uber,
  `tel:`, contact resolution), returns the contract JSON (§5), writes history.
- Permissions: Contacts, Reminders. Request them from the app's Setup screen,
  not mid-intent.
- `AppShortcutsProvider` with a phrase like "Relay" so it also works from Siri.

Acceptance: from the Shortcuts app, Decide returns correct JSON for the eval
utterances; native modules work.

### Phase 4 — Router + module shortcuts

- Specs for the router and every v1 shortcut module (§6, §9).
- User builds them; agent exports and commits XML sources (if on a Mac).
- Confirmation: router shows an alert with `summary` when `confirm` is true.
- `respond` / `clarify` / `done` / `error` → Speak Text.

Acceptance: every v1 module works by voice on the device, triggered from the
Action Button or Back Tap.

### Phase 5 — Tune

- Run evals, read History from real use, adjust module descriptions and the
  system prompt. Track accuracy over time in `evals/RESULTS.md`.
- Add utterances for every real failure seen in History.

### Later (not v1)

- More native App Intents as building blocks (HTTP request, clipboard,
  calendar events, HomeKit).
- Multi-turn `clarify` (router re-dictates and calls Decide with context).
- Generating new shortcut plists from the XML sources in `shortcuts/src/`.
- User-registered custom shortcuts in the registry.
- TestFlight: onboarding for installing the router via iCloud link, privacy
  policy (states: no data collected; transcripts go only to the chosen provider).

---

## 11. Rules for the executing agent

- Work only inside `relay/`. Don't read from, import from, or mimic other
  projects in this repo.
- Never commit, log, or print API keys. Use the Keychain in the app and env
  vars for `relay-eval`.
- Don't add third-party dependencies without asking.
- Verify Apple API details (App Intents return types, Shortcuts action names,
  `shortcuts sign` flags) against current documentation or the installed SDK
  instead of trusting this plan. Where reality differs, follow reality and note
  it in `relay/DECISIONS.md`.
- Stop and ask the user after Phase 0, and whenever a design assumption here
  turns out to be false.
