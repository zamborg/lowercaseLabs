# Transcription Upgrade: Live Streaming via AVAudioEngine

## Background

theVoid currently transcribes audio **after** recording stops:

1. User records → stops → WhisperKit processes the `.m4a` file
2. Transcript appears in an editable sheet (~3–5s wait on A15+)
3. User edits if needed → submits

This works well. The goal of this upgrade is to show a **live, updating transcript while the user is still speaking**, eliminating the post-recording wait entirely.

---

## How blackhole does it (reference implementation)

blackhole's `AudioCapturePipeline` + `StreamingWhisperSession` demonstrate the pattern:

1. **`AVAudioEngine` input tap** captures raw 16 kHz mono PCM audio into a growing `[Float]` accumulation buffer.
2. Every **~500 ms of new audio**, a snapshot of the full accumulated buffer is passed to `whisperKit.transcribe(audioArray:decodeOptions:)`.
3. The UI swaps in the latest result — text refines with each pass as more audio context is available.
4. On stop, one final full-buffer transcription is run (`isFinal: true`).

WhisperKit has **no streaming API**. The "streaming" effect is achieved by repeatedly re-transcribing the growing buffer. Because `tiny.en` on A15+ processes ~10 s of audio in ~1 s, a 500 ms update stride keeps up comfortably in real time.

Key file: `blackhole/ios/Sources/Dictation/StreamingWhisperSession.swift`

---

## What needs to change in theVoid

### Problem

theVoid uses `AVAudioRecorder` (writes directly to an `.m4a` file). `AVAudioEngine` and `AVAudioRecorder` can't share the same `AVAudioSession` tap simultaneously.

### Solution: port `RecorderEngine` from `AVAudioRecorder` → `AVAudioEngine`

`AVAudioEngine` can do everything `AVAudioRecorder` does today, plus expose raw PCM buffers:

| Responsibility | Current | After |
|---|---|---|
| Write `.m4a` file | `AVAudioRecorder` | `AVAudioEngine` + `AVAudioFile` |
| Feed PCM to WhisperKit | ❌ | `AVAudioEngine` input tap |
| Amplitude metering | `AVAudioRecorder.averagePower` | manual RMS on tap buffers |
| Session config | `AVAudioRecorder` | manual `AVAudioSession` setup |

### New architecture

```
AVAudioEngine
  └── inputNode
        └── tap (16 kHz, 1024 samples/buffer, ~64 ms)
              ├── PCMBufferSampleConverter → [Float] accumulation buffer
              │     └── StreamingWhisperSession (throttled at 500 ms stride)
              │           └── WhisperTranscriptionRuntime.shared.transcribe(audioArray:)
              │                 └── publishes AppModel.liveTranscript (updates TextEditor live)
              └── AVAudioFile (.m4a) — same file path as today, iCloud sync unchanged
```

### Files to create / modify

| File | Change |
|---|---|
| `RecorderEngine.swift` | Replace `AVAudioRecorder` with `AVAudioEngine`; add file writing via `AVAudioFile`; expose PCM tap |
| `AudioCapturePipeline.swift` (new) | Port from blackhole — manages AVAudioEngine setup, session config, buffer tap |
| `StreamingWhisperSession.swift` (new) | Port from blackhole — accumulation buffer, 500 ms stride, partial/final transcription |
| `VoiceTranscription.swift` | Add `transcribe(audioArray: [Float], ...)` overload alongside existing file-based method |
| `AppModel.swift` | `beginTranscriptionForReview` becomes a no-op (transcript already populated by the time recording stops) |
| `ViewsVoid.swift` | Remove the post-stop transcription spinner — transcript is ready immediately |

### What stays the same

- `.m4a` file written to `DraftStore` — path, iCloud sync, playback: unchanged
- `WhisperTranscriptionRuntime.shared` (actor singleton) — reused, add `audioArray` overload
- `TranscriptReviewSheet` — still shows editable text, just pre-populated without a wait
- Model download flow — unchanged

---

## Trade-offs

| | Live streaming | Current (post-stop) |
|---|---|---|
| UX | Text visible while speaking | ~3–5s wait after stop |
| Complexity | High — AVAudioEngine rewrite | Low |
| Battery / CPU | Higher (continuous inference) | Low (one-shot) |
| Accuracy | Same model, same quality | Same |
| Older devices | May lag on A13 and below | Fine |

---

## Implementation order

1. Port `AudioCapturePipeline` from blackhole (thin wrapper — mostly a copy)
2. Port `StreamingWhisperSession` from blackhole (same)
3. Add `transcribe(audioArray:)` to `WhisperTranscriptionRuntime`
4. Rewrite `RecorderEngine` to use `AVAudioEngine` + `AVAudioFile`, wiring the tap to `StreamingWhisperSession`
5. Update `AppModel` — remove `beginTranscriptionForReview` task, wire `liveTranscript` to session updates
6. Update `ViewsVoid` — `TranscriptReviewSheet` opens immediately on stop (no spinner)
7. Delete old `SFSpeechRecognizer` permission handling if it crept back in
