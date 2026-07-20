# Streaming Transcription Working Doc

Status: implemented; physical-device validation pending
Last updated: 2026-07-15

## Goal

Make recording feel live by showing an updating transcript while the user is speaking, then open the review sheet with text already populated.

This is a UX upgrade, not a backend change. It preserves the local-first model:

- Audio drafts still save locally.
- Final entries still keep playable audio.
- iCloud sync for drafts/audio/entries remains unchanged.
- Apple SpeechAnalyzer is the iOS 26 default; WhisperKit remains an on-device alternative.
- Liquid insights still run after the final transcript is accepted/submitted.

## Current Flow

Relevant files:

- `theVoid/theVoid/AudioAndDraft.swift`
- `theVoid/theVoid/StreamingTranscription.swift`
- `theVoid/theVoid/VoiceTranscription.swift`
- `theVoid/theVoid/AppModel.swift`
- `theVoid/theVoid/ViewsVoid.swift`

Current path:

1. `RecorderEngine` uses one `AVAudioEngine` input tap.
2. The tap writes linear PCM to a temporary `.caf`, updates the amplitude meter, and feeds the selected live session. WhisperKit converts buffers to 16 kHz mono samples; Apple converts them to `SpeechAnalyzer.bestAvailableAudioFormat`.
3. The selected `LiveTranscriptionSession` publishes provisional text during capture. Apple uses `SpeechAnalyzer`; WhisperKit starts after 0.4 seconds and refreshes after each additional second of audio.
4. On stop, the review sheet opens immediately with the latest partial while transcription finalization and Apple M4A export run concurrently.
5. The final text replaces the partial unless the user has already edited it.
6. If streaming or the final sample pass fails, the existing file-based transcription path retries from the saved `.m4a`.
7. Submit still calls `submitDraft(... overrideTranscript:)`, so Liquid analysis reuses the accepted transcript.

## Blackhole Reference

Blackhole already has a useful implementation:

- `../blackhole/ios/Sources/Dictation/DictationEngine.swift`
- `../blackhole/ios/Sources/Dictation/AudioCapturePipeline.swift`
- `../blackhole/ios/Sources/Dictation/StreamingWhisperSession.swift`
- `../blackhole/ios/Sources/Dictation/PCMBufferSampleConverter.swift`
- `../blackhole/ios/Sources/Dictation/WhisperKitDictationEngine.swift`

Pattern:

1. `AVAudioEngine` installs an input tap.
2. Tap buffers are converted to 16 kHz mono `Float` samples.
3. A session accumulates samples.
4. Every stride, it retranscribes the accumulated sample array using `WhisperKit.transcribe(audioArray:)`.
5. The latest result replaces the visible text.
6. On stop, it runs one final transcription over the full sample buffer.

Important constraint: WhisperKit is not doing true token streaming for speech. This is repeated partial transcription over a growing buffer.

## Implemented Architecture

Keep the `theVoid` product flow, but split recording into two outputs:

```text
Microphone
  -> AVAudioEngine input tap
      -> amplitude meter for the dot UI
      -> PCM buffers for the selected LiveTranscriptionSession
      -> durable audio writer for existing draft/audio storage
```

The engine boundary lives in `TranscriptionEngine.swift`:

- `TranscriptionEngine` prepares a language, creates a live session, transcribes a file, and unloads its resources.
- `LiveTranscriptionSession` accepts PCM buffers and exposes provisional/final text.
- `TranscriptionEngineCoordinator` owns exactly one selected engine and swaps implementations when settings change.
- `TranscriptionConfiguration` is the persisted engine/language pair.

Implementations:

- `AppleSpeechTranscriptionEngine` uses iOS 26 `SpeechAnalyzer` and `SpeechTranscriber` with progressive volatile/final results. It converts hardware input to Apple's compatible analyzer format, and its input queue is bounded to prevent microphone buffers from accumulating under pressure.
- `WhisperKitTranscriptionEngine` owns an isolated Whisper runtime so replacing one engine cannot unload another operation's model.
- `StreamingWhisperSession` provides repeated partial Whisper inference plus one accurate final pass.

Both live sessions provide:

- Partial transcript updates while recording.
- Final transcript after stop.
- Silent partial-failure tolerance so audio capture continues and file transcription can take over.

`RecorderEngine` remains the UI-facing object, with `AVAudioEngine` replacing `AVAudioRecorder` internally.

## Audio Persistence Decision

The implementation writes the microphone's linear PCM format to a temporary `.caf` with `AVAudioFile`. After capture stops, `AVAssetExportSession` converts that file to the existing draft `.m4a` while the selected engine finalizes transcription concurrently.

This preserves the existing storage contract while keeping codec negotiation out of recording startup:

- Keeps the existing draft, playback, journal, and iCloud file contract unchanged.
- Lets the review sheet open immediately from the live partial transcript while export completes.
- Uses Apple's M4A export path instead of asking an AAC writer to accept a live hardware format.
- Avoids running `AVAudioRecorder` and `AVAudioEngine` at the same time.

Direct AAC writing was rejected on physical hardware with `kAudioCodecUnsupportedFormatError` (`!dat`). PCM capture plus M4A export is covered by a focused simulator test; microphone behavior still needs a physical-device retest.

## Implemented Components

- `TranscriptionEngine.swift` owns the implementation-neutral protocols, engine/language preferences, and coordinator.
- `AppleSpeechTranscription.swift` owns iOS 26 SpeechAnalyzer asset preparation, live analysis, final file analysis, and result reconciliation.
- `StreamingTranscription.swift` owns Whisper PCM conversion, bounded partial scheduling, rolling transcript reconciliation, and the final sample pass.
- `WhisperTranscriptionRuntime` supports file, full-sample, and fast plain-text partial transcription with a prewarmed language-appropriate model.
- `RecorderEngine` owns the engine tap, PCM writer, M4A export, metering, warnings, auto-stop, live text, and finalization task.
- `VoidExperienceView` shows the latest transcript tail during recording and opens review immediately on stop.
- `TranscriptReviewSheet` shows text while the final pass is running and preserves user edits if they begin before final reconciliation.
- `AppModel` retains file transcription as the fallback and the accepted-transcript seam for Liquid analysis.

## Engine And Language Settings

Settings exposes two controls:

- Engine: Apple or WhisperKit. Apple is the default on iOS 26 and WhisperKit is the fallback on earlier supported OS versions.
- Language: device language or a specific supported locale.

For WhisperKit, English locales select `tiny.en`; every other locale selects multilingual `tiny` and passes its language code into decoding. Changing model families may require a one-time model download. Apple resolves the requested locale against `SpeechTranscriber` support and installs/reserves the matching system speech asset.

## Whisper Memory Policy

The original growing-buffer approach repeatedly decoded the entire recording and eventually caused memory pressure. Live decoding is now bounded while the full pass after stop remains the source of truth.

Policy:

- Start partial transcription after 0.4 seconds of audio.
- Run at most one partial transcription at a time.
- Queue the newest audio whenever another 1 second is available during inference.
- Decode at most the latest 15 seconds for each live update.
- Merge each rolling window into already displayed text using normalized word overlap.
- Use the lower-cost plain-text decode path for live updates; reserve the accurate full pass for stop.
- Always run one final transcription at stop.

At the 2:30 recording limit, retained 16 kHz mono samples are about 9.6 MB. Live inference input stays capped at about 0.96 MB rather than growing to the full recording on every pass.

## UX Notes

Recording screen:

- Show partial transcript below the timer or in a compact live preview.
- Keep the primary touch target unchanged.
- Do not make the user manage transcription state.

Review sheet:

- Remove the default spinner when final text is already available.
- Keep spinner only for fallback final transcription.
- Preserve edit-before-share behavior.

Error handling:

- If live partials fail, do not interrupt recording.
- If final transcription fails, keep the audio draft and show the existing retry/retranscribe path.

## Known Risks

- `AVAudioEngine` rewrite can regress recording reliability.
- PCM capture and post-stop M4A export still need interruption and Bluetooth-route testing.
- Repeated WhisperKit inference can raise battery/thermal cost.
- Apple SpeechAnalyzer requires iOS 26 and a supported downloadable language asset.
- The newest words can revise as Whisper gains context; the UI distinguishes that active tail from older text.
- Simulator microphone behavior is not enough; this needs device testing.

## Release Checklist

- [x] Build with Xcode 26.6 / iOS 26.5 SDK.
- [x] Unit tests pass on iPhone 17 Pro / iOS 26.5 Simulator (22 passed, 0 failed), including 0.4-second live callback, 15-second inference bound, rolling-window merge, language/model selection, provider attribution, and PCM-to-M4A playback validation.
- [x] Partial-text reconciliation has focused unit coverage.
- [x] Apple and WhisperKit implementations compile behind the same engine/session interfaces with Xcode 26.6.
- [ ] Recording starts/stops reliably on device.
- [ ] Lock recording still works.
- [ ] Auto-stop still works.
- [ ] Live partials appear within a few seconds.
- [ ] Apple volatile/final text behaves correctly for a full 2:30 device recording.
- [ ] WhisperKit stays within acceptable memory/thermal limits for a full 2:30 device recording.
- [ ] Switching engine and language downloads/prepares the correct assets and persists after relaunch.
- [ ] Final transcript appears without post-stop wait on normal devices.
- [ ] Fallback file transcription still works.
- [ ] Draft deletion still works.
- [ ] Saved entry playback still works.
- [ ] iCloud draft/audio sync still works.
- [ ] 2:30 recording does not overheat or stall.
