# Controllr Voice 1.6

Run **Controllr Voice 1.6** directly in Shortcuts. The optional shortcut input is a thread UUID, used by **Continue by voice** in Controllr. The app retains the full thread and uses up to the 48 most recent entries for model context.

1. Call **Decide** with the dictation, current thread ID, and `routerVersion: 3`. Store the returned `thread_id`.
2. On `run`, use the returned `confirmation` value. If it is `ask`, show Continue/Cancel and record cancellation if selected. Call **Authorize Controllr Action** with the thread and action IDs before dispatch. Controllr checks whether the action is enabled, verifies approval, and starts the action once.
3. Dispatch the returned `shortcut` with its `input`. The dispatcher uses shared completion paths for actions with and without text output. Both call **Complete Controllr Action** with the same thread and action IDs. For a staged native write, the callback performs the write. For a Shortcut read, it appends a tool result, resumes the model turn, and returns another decision.
4. On `voice_next: listen`, finish speaking before starting the next dictation. On `stop`, speak and exit. On `silent`, exit. Saying “cancel,” “stop,” or “that’s all” ends the session. The shortcut allows 32 iterations across dictation and action results. Controllr can resume the thread afterwards.

A Shortcut step receipt confirms that the workflow reached the callback. A phone call, message delivery, or ride booking can have its own outcome after the handoff.

## Model turn

The selected provider receives the conversation and the available tools. Anthropic uses the Claude Messages API, OpenRouter uses Chat Completions, and OpenAI uses Responses for reasoning models and Chat Completions for older chat models. Each adapter returns assistant text, one tool call, or both. Ordinary final text ends the voice session. Reasoning levels are set per model under **Setup**.

The `clarify` tool asks a spoken question. The app saves the assistant text and tool call in the thread and sends the next utterance back as the result of that call. A direct question emitted as plain assistant text also keeps the shortcut listening. Lookup results return to the model during the same turn. Users set per-action enablement and confirmation under **Modules**.

## Build and install

Generate the Xcode project with `xcodegen generate` from `controllr/` and build the Controllr scheme with your Apple Development team. Generate and sign the shortcut with Python 3.12 and Shortcuts Playground. Set `SHORTCUTS_PLAYGROUND_DIR` to the directory containing `scripts/validate_shortcut.py` and `scripts/sign_shortcut.sh`.

```sh
uv run --python 3.12 --no-project shortcuts/gen.py --team-id 9G57J2JMF7
plutil -lint 'shortcuts/src/Controllr Voice 1.6.xml'
uv run --python 3.12 --no-project "$SHORTCUTS_PLAYGROUND_DIR/scripts/validate_shortcut.py" \
  --target-platform all --target-macos 27 'shortcuts/src/Controllr Voice 1.6.xml'
"$SHORTCUTS_PLAYGROUND_DIR/scripts/sign_shortcut.sh" \
  'shortcuts/src/Controllr Voice 1.6.xml' --name 'Controllr Voice 1.6' --output-dir shortcuts/signed
```

The combined platform catalog accounts for Clock actions that the plugin classifies as Mac-only despite the earlier iPhone checks. The validator can flag the Send Message body token shape described in [STATUS](../../STATUS.md). Check the actual editor and phone before changing the Messages binding.

Import the signed shortcut in macOS Shortcuts and let it sync to the phone. Start with a spoken answer or a clarification that you cancel. Use Controllr's Threads tab to check the transcript.

## Add an action

1. Add a manifest in [Actions](../../ControllrCore/Sources/ControllrCore/Actions), named after the action ID. Give it parameters, a summary, an operation of `read`, `change`, or `handoff`, any required permissions, the default confirmation setting, and at least one eval. The registry supplies the model schema and Modules page across providers.
2. Add an app-side case in `App/NativeExecutor.swift` or assign a `shortcutHandler` in `shortcuts/gen.py`. Native reads return `.toolResult`, native writes return `.spoken`, and app handoffs return `.openURL`. A Shortcut read handler returns a text reference. Rebuild the app after changing manifests. Regenerate and sign the shortcut when its dispatch changes.
3. Run `swift test --package-path ControllrCore` and `swift run --package-path ControllrCore controllr-eval --provider openai --model YOUR_MODEL`. The default eval input is the action manifests. Add `--file PATH` to run an external JSONL suite.

Pass `--provider anthropic` with `ANTHROPIC_API_KEY` or `--provider openrouter` with `OPENROUTER_API_KEY` to run the same suite through either adapter. Use `--list-models` to check the catalog available to that key and `--reasoning low` to test an explicit effort.
