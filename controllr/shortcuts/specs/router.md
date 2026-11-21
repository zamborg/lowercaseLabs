# Controllr Voice

Run Controllr Voice directly in Shortcuts. The optional shortcut input is a thread UUID, used by **Continue by voice** in Controllr. The app retains the full thread and uses up to the 48 most recent entries for model context.

1. Dictate the user turn and call Controllr's **Decide** action with the text and current thread ID.
2. Save the returned `thread_id` locally in the shortcut. On `clarify`, speak the question and dictate again in the same thread. Saying “cancel” or “stop” ends the conversation. After eight clarification turns, the thread can continue from Controllr.
3. On `run`, use the returned `shortcut` and `input`. Show Continue/Cancel for actions whose manifest requires confirmation. Write the step's completion or cancellation to the thread using **Record Controllr Action** and exit. A cancelled or failed system dialog can leave the step without a receipt.
4. For an answer, a completed app-side action, or an error, speak the summary and exit.

## Build and install

Generate the Xcode project with `xcodegen generate` from `controllr/` and build the Controllr scheme with your Apple Development team. Generate and sign the shortcut with Python 3.12 and Shortcuts Playground. Set `SHORTCUTS_PLAYGROUND_DIR` to the directory containing `scripts/validate_shortcut.py` and `scripts/sign_shortcut.sh`.

```sh
uv run --python 3.12 --no-project shortcuts/gen.py --team-id 9G57J2JMF7
plutil -lint 'shortcuts/src/Controllr Voice.xml'
uv run --python 3.12 --no-project "$SHORTCUTS_PLAYGROUND_DIR/scripts/validate_shortcut.py" \
  --target-platform all --target-macos 27 'shortcuts/src/Controllr Voice.xml'
"$SHORTCUTS_PLAYGROUND_DIR/scripts/sign_shortcut.sh" \
  'shortcuts/src/Controllr Voice.xml' --name 'Controllr Voice' --output-dir shortcuts/signed
```

The combined platform catalog accounts for Clock actions that the plugin classifies as Mac-only despite the earlier iPhone checks. The validator can flag the Send Message body token shape described in [STATUS](../../STATUS.md). Check the actual editor and phone before changing the Messages binding.

Import the signed shortcut in macOS Shortcuts and let it sync to the phone. Start with a spoken answer or a clarification that you cancel. Use Controllr's Threads tab to check the transcript.

## Add an action

1. Add a manifest in [Actions](../../ControllrCore/Sources/ControllrCore/Actions), named after the action ID. Give it parameters, a summary, the confirmation rule, and one or more eval cases. Eval cases can specify earlier turns and selected expected arguments.
2. Implement an app-side action in `App/NativeExecutor.swift` or assign a `shortcutHandler` implemented in `shortcuts/gen.py`. Native actions that need to open another app return a URL for the shortcut to handle. Rebuild the app after changing the manifests and regenerate the shortcut after changing dispatch.
3. Run `swift test --package-path ControllrCore` and `swift run --package-path ControllrCore controllr-eval --provider openai --model YOUR_MODEL`. The default eval input is the action manifests. Add `--file PATH` to run an external JSONL suite.
