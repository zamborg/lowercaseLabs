
# controllr build plan

> Original spike plan. See [STATUS.md](STATUS.md) and [the Controllr Voice spec](shortcuts/specs/router.md) for the current build.

controllr is a voice-first "better Siri" for iPhone: one router shortcut dictates a request, our App Intent asks an LLM (the user's own OpenAI or Anthropic key) to pick a module, and the router runs it. This is the full execution plan for an agent building it. Read it end to end, run the phases in order, and stop after Phase 0 to report results.

## 1. What we're building
The user triggers controllr (Action Button, Back Tap, Control Center, Siri phrase), says something, and it does the thing:
"Text Sam I'm running ten minutes late"

"Skip this song"

"Get me an Uber home"

"Remind me to call the dentist tomorrow at 9"

"What's 18% of 64?" (just answers)

A single router shortcut dictates the speech and hands the transcript to our app's Decide App Intent. Decide asks an LLM to choose one module from a known registry and fill in its arguments. The router then runs that module.

### Principles
Local-only. No backend, no analytics, no accounts. The only network calls go to the LLM provider the user configured.

Bring your own key. The user pastes an OpenAI and/or Anthropic API key. It is stored in the iOS Keychain and never logged or committed.

We own the registry. The app ships knowing its modules (id, description, input schema, shortcut name). It never tries to read the user's shortcut library.

Fast by default, confirm when it matters. Harmless actions fire immediately. Anything that sends, buys or calls shows a confirmation.

Clean slate. Do not copy code, patterns, names or structure from any other project in this repository. Everything lives under controllr/.


### Out of scope for v1
App Store onboarding polish, LLM-generated shortcuts, multi-turn conversation, on-device models, and iPad, Mac or Watch targets.

## 2. Architecture
The shortcut is the entry point and the app is a background "brain" action, so everything stays inside Shortcuts with no app switching.
The router shortcut, step by step:
Dictate Text captures the request.

Controllr: Decide (our App Intent, input = transcript) calls the LLM and returns JSON text (§5).

Get Dictionary from Input parses it.

If action is run: show an alert with summary when confirm is true (Cancel stops), then Run Shortcut named by shortcut with input as its input.

Otherwise (done, respond, clarify, error): Speak Text with speak.

Why this shape: a third-party app can't silently run a user's shortcuts. The only API is the shortcuts:// URL, which brings the Shortcuts app to the front. System capabilities our code can't call (Send Message, Skip Forward, Start Timer, Set Focus) live in small module shortcuts; things our code can do live as native modules.

### Two kinds of modules
Kind
Executed by
Examples
Adding one requires

Shortcut module
A module shortcut run by the router
send message, skip track, start timer, set focus, flashlight
A registry entry + a new shortcut on the phone

Native module
Our app, inside the Decide intent
open a URL or deep link (Uber, Maps), add a reminder (EventKit), place a call
A registry entry + Swift code + an app rebuild

Long-term, prefer native modules: they're testable and tunable in Swift, and the router never changes.

## 3. Tech choices
Swift 6 / SwiftUI, iOS deployment target 17.0. The user's device is an iPhone 14; confirm its installed iOS version and test there.

App Intents for the Decide intent and native modules. Intents live in the main app target, not an extension, so they share the app's Keychain and storage with no App Group setup. Set static let openAppWhenRun = false.

XcodeGen (project.yml) generates the Xcode project. Never hand-edit .pbxproj; regenerate instead.

Swift package ControllrCore holds all logic that doesn't need UIKit: registry, prompt building, LLM clients, response parsing and the decision contract. It also builds on macOS, so it can be unit-tested and evaluated from the command line without a phone.

Networking: plain URLSession + Codable. No third-party SDKs.

Persistence: SwiftData for history, the Keychain (Security framework behind a small wrapper) for API keys, UserDefaults for non-secret settings.

Speech: the Shortcuts Dictate Text action (on-device, free). The app does not record audio itself in v1.


## 4. Repository layout
Everything lives in controllr/ in the lowercaseLabs repo. This plan is also committed there as controllr/PLAN.md.
Add controllr/shortcuts/signed/, any generated .xcodeproj, and any .env or key files to .gitignore.

## 5. Decision contract (Decide → router)
The Decide intent returns a JSON string, which the router parses with "Get Dictionary from Input". Keep it flat and stable: the router depends on it.
action
When
Router does

run
A shortcut module was chosen
Confirm if confirm, then Run Shortcut shortcut with input

done
A native module already ran inside Decide
Speak speak

respond
The answer module: a question or calculation
Speak speak

clarify
The request is ambiguous or missing required info
Speak speak and end (v1)

error
Network, key or parsing failure
Speak speak

Rules:
run is only for shortcut modules. Native modules execute inside the intent and return done.

confirm comes from the module's registry entry (requiresConfirmation), never from the LLM. The LLM can't skip a confirmation.

Every response has either shortcut or speak, so the router never has to guess.

clarify ends the run in v1. Multi-turn comes later.


## 6. Registry
The registry is code in ControllrCore, versioned with the app. Each module is also an LLM tool.

### v1 modules
id
Kind
Input
Confirm
Notes

send_message
shortcut "Controllr · Send Message"
recipient, body
yes
Resolve the name to a phone number in-app with the Contacts framework before returning. 0 or 2+ matches → clarify.

call_contact
native (tel: URL)
recipient
yes
Same contact resolution.

media_control
shortcut "Controllr · Media"
command: play / pause / next / previous
no
System Play/Pause and Skip actions; works with whatever is playing, Spotify included.

start_timer
shortcut "Controllr · Timer"
minutes, seconds
no

set_alarm
shortcut "Controllr · Alarm"
time (HH:mm), label
no

add_reminder
native (EventKit)
title, due (ISO 8601, optional)
no
The prompt includes the current date, time and time zone.

create_note
shortcut "Controllr · Note"
text
no

directions
native (Apple Maps URL)
destination, mode: drive / walk / transit
no
"home" resolves to the saved home address.

request_ride
native (Uber deep link)
destination
yes
Opens Uber with the destination filled in; the user confirms in Uber. Check the deep-link format against Uber's current docs.

set_focus
shortcut "Controllr · Focus"
focus: dnd / sleep / work / off
no

flashlight
shortcut "Controllr · Flashlight"
on: bool
no

answer
built-in
text
no
General questions, math, quick facts → respond.

clarify
built-in
question
no
The request is ambiguous or missing required info.

Settings supply the home address and an optional nickname map ("mom" → a contact).

## 7. LLM routing
One tool per module. Tool name = module.id, parameters = module.input. Force a tool call: OpenAI tool_choice: "required", Anthropic tool_choice: {"type": "any"}. Use exactly one call and ignore extras.

System prompt (built in ControllrCore/Prompt), kept short:
You are the router for a voice assistant on the user's iPhone.

Current date, time, time zone and locale.

Pick exactly one tool. Prefer clarify over guessing a recipient or destination.

The transcript comes from dictation and may contain recognition errors; infer the intent.

For answer, reply in one or two spoken sentences.


Providers: protocol LLMProvider { func route(system:, transcript:, tools:) async throws -> ToolCall }.
OpenAIProvider: Chat Completions or Responses API with tools. Pick one and stay consistent.

AnthropicProvider: Messages API with tools (x-api-key and anthropic-version headers).


Models: never hard-code a guessed model name. The model is a user setting. Fill the picker from each provider's list-models endpoint using the user's key. For testing the user wants a current small, fast OpenAI model; for Anthropic, default to claude-haiku-5-5.

Latency budget: under about 2 s from the end of dictation to the action. Use a small model, low max_tokens, a 10 s request timeout, and no retries in the intent path, since App Intents have limited background run time.

Validation: decode the tool arguments against the module schema. On failure, return error with a short spoken message and log the raw response to History.


## 8. App UI
Minimal and functional: four tabs or one list-based screen.
Setup: provider picker; secure API key fields saved to the Keychain, with a "Test key" button that calls list-models; model picker; home address; a router setup checklist (link to the router shortcut, how to bind it to the Action Button or Back Tap).

Modules: the registry, showing each module's description, inputs and whether it needs confirmation. For shortcut modules, a checklist the user ticks once the shortcut is installed.

History: every run, with transcript, provider and model, chosen module and arguments, decision JSON, latency and any error. Tap to copy; allow "Clear history". This is the main tuning tool.

Try it: a text field that runs the same Router as the intent, without executing shortcut modules, so routing can be tested in the app without dictation.


## 9. Shortcuts: authoring and installing
The router and module shortcuts are built by hand once, then kept in the repo as diffable sources.
Write an exact spec for each shortcut in shortcuts/specs/<name>.md: the ordered action list, every parameter and every variable.

The user builds it in the Shortcuts app from the spec.

On a Mac, export the shortcut, convert it to XML with plutil -convert xml1 <file>, and commit it to shortcuts/src/. This also reveals the real action identifiers for writing future shortcuts.

To push a new or edited shortcut, make sign runs shortcuts sign --mode anyone --input src/X.shortcut --output signed/X.shortcut. This is macOS only and needs the Mac signed into iCloud. Open the signed file on the Mac to import it; it syncs to the iPhone through iCloud. AirDrop also works.

Module shortcut convention: the input is a dictionary. The first action is "Get Dictionary from Input", followed by one "Get Dictionary Value" per field. Module shortcut names match the registry's shortcut names exactly.

## 10. Phases
Run the phases in order. Commit at the end of each with a clear message, and don't move on until its acceptance criteria are met.

### Phase 0: Spike (gate)
Prove the assumptions the design depends on before building anything else.
A minimal app target with one DecideIntent (openAppWhenRun = false) that returns a hard-coded JSON string, and a variant that reads a value from the Keychain and makes a real HTTPS request.

A router shortcut: Dictate Text → Decide → Get Dictionary from Input → Run Shortcut, using the shortcut value from the dictionary as the shortcut to run → a test module that shows an alert with its input.

Acceptance (report each to the user as pass or fail):
Run Shortcut by variable name works. If not, the fallback is a router built from an If / Otherwise If chain that matches shortcut against each known module name. Report which approach is in use.

Decide runs from the shortcut without opening the app, reads the Keychain, and completes a network call.

End-to-end latency on the device, from the end of dictation to the test module.

It works from the Lock Screen and the Action Button, or note what needs the phone unlocked.

Stop here and report to the user before continuing.

### Phase 1: ControllrCore and evals
Module, InputSchema, Decision, and the built-in v1 registry (§6).

Prompt builder, OpenAIProvider, AnthropicProvider, Router.

Unit tests with a mocked provider: schema → tool JSON, tool call → Decision, confirmation comes from the registry, invalid arguments → error.

controllr-eval CLI: reads evals/utterances.jsonl, takes the key from an environment variable, and prints accuracy per module and latency. Never print the key. Write about 40 utterances covering every module, ambiguous cases that should clarify, and dictation-style errors.

Acceptance: swift test passes, and controllr-eval runs against a real key and reports results.

### Phase 2: App shell
XcodeGen project, Keychain wrapper, settings, the Setup / Modules / History / Try it screens (§8), and the SwiftData history model.

Acceptance: builds and runs on the device; keys persist; "Test key" works; "Try it" shows the routed decision and records it in History.

### Phase 3: Decide intent, for real
DecideIntent calls Router, runs native modules (Reminders, Maps, Uber, tel:, contact resolution), returns the contract JSON (§5), and writes History.

Request Contacts and Reminders permissions from the Setup screen, never mid-intent.

AppShortcutsProvider with a phrase like "Controllr" so it also works from Siri. Siri hears the name as "controller": test that the phrase is recognized, and add an easy-to-say alternate if not.

Acceptance: run from the Shortcuts app, Decide returns correct JSON for the eval utterances, and native modules work.

### Phase 4: Router and module shortcuts
Specs for the router and every v1 shortcut module (§6, §9).

The user builds them; the agent exports and commits the XML sources when on a Mac.

Confirmation: the router shows an alert with summary when confirm is true.

respond, clarify, done and error → Speak Text.

Acceptance: every v1 module works by voice on the device, triggered from the Action Button or Back Tap.

### Phase 5: Tune
Run evals, read History from real use, and adjust module descriptions and the system prompt. Track accuracy over time in evals/RESULTS.md.

Add an utterance for every real failure seen in History.


### Later (not v1)
More native App Intents as building blocks: HTTP request, clipboard, calendar events, HomeKit.

Multi-turn clarify: the router dictates again and calls Decide with context.

Generating new shortcut plists from the XML sources in shortcuts/src/.

User-registered custom shortcuts in the registry.

TestFlight: onboarding that installs the router from an iCloud link, and a privacy policy (no data collected; transcripts go only to the chosen provider).


## 11. Rules for the executing agent
Work only inside controllr/. Don't read from, import from, or imitate other projects in this repo.

Never commit, log or print API keys. Use the Keychain in the app and environment variables for controllr-eval.

Don't add third-party dependencies without asking.

Check Apple API details (App Intents return types, Shortcuts action names, shortcuts sign flags) against current documentation or the installed SDK instead of trusting this plan. Where reality differs, follow reality and record it in controllr/DECISIONS.md.

Stop and ask the user after Phase 0, and whenever an assumption in this plan turns out to be false.

