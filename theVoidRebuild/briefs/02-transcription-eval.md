# Brief 02: transcription eval (Codex)

Swift package `eval/transcription/` (macOS 26, SwiftPM executable `wer-eval`).

- Input dir of audio files + `reference.jsonl` ({file, text}).
- Run each file through SpeechAnalyzer + SpeechTranscriber (on device locale en-US), finalized results only.
- Normalize (lowercase, strip punctuation) and compute WER via word-level Levenshtein; also report time-to-final per file and real-time factor.
- Output `wer.md` and `results.json`. Unit-test the WER function.
Done = `swift run wer-eval --audio <dir> --ref reference.jsonl` prints aggregate WER.
