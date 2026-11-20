# Transcription evaluation

A dependency-free macOS 26 SwiftPM executable using Apple's on-device
`SpeechAnalyzer` and `SpeechTranscriber`, fixed to en-US with the `.transcription`
preset. It evaluates finalized results only. Requires a macOS 26 SDK, Swift 6.2
or later, compatible hardware, and the system's en-US transcription assets.

Run all commands below from `eval/transcription/`.

```sh
swift test
python3 scripts/make_synthetic.py
swift run wer-eval --audio fixtures/synthetic/audio --ref fixtures/synthetic/reference.jsonl
```

The final command prints aggregate WER and writes `wer.md` and `results.json`
in the current directory. These default reports are the synthetic smoke test.
Use a release build for meaningful timing comparisons:

```sh
swift run -c release wer-eval --audio fixtures/synthetic/audio --ref fixtures/synthetic/reference.jsonl --output reports/synthetic
```

The default command ensures the en-US model is ready, downloading/installing
system-managed assets if needed. To fail instead when installation is required:

```sh
swift run wer-eval --audio fixtures/synthetic/audio --ref fixtures/synthetic/reference.jsonl --no-install-assets
```

Initial asset status is recorded before reservation/installation. The system
may return `supported` on a later invocation even after a successful prior run;
setup always obtains an installation request to check readiness for the current
executable, rather than relying only on that status.
Asset installation may need network access; it is measured separately and
excluded from transcription timing. Audio processing uses `SpeechTranscriber`
on device. No legacy recognizer or server fallback is used. This tool reads
files and does not access the microphone. Apple's [permission documentation](https://developer.apple.com/documentation/speech/asking-permission-to-use-speech-recognition)
distinguishes `SpeechAnalyzer` from the `SFSpeechRecognizer` authorization flow;
this tool does not request legacy speech-recognition permission. System policy,
file permissions, model downloads, and speech services can still fail. Run from
a normal Terminal session if an execution sandbox prevents system-service access.
Failures are reported as failures, never fabricated hypotheses.

## Future recordings

Put the owner's roughly 20 clips in an audio directory, and hand-correct one
transcript per clip in a UTF-8 `reference.jsonl`:

```jsonl
{"file":"clip-01.m4a","text":"The actual words spoken in the clip.","source":"recorded"}
{"file":"clip-02.wav","text":"Another corrected transcript.","source":"recorded"}
```

```sh
swift run -c release wer-eval --audio /path/to/audio --ref /path/to/reference.jsonl --output reports/recorded
```

The minimal `{file,text}` format also works; missing `source` is reported as
`unspecified`. Supported source labels are `synthetic`, `recorded`, and
`unspecified`. Keep synthetic and recorded runs separate. Files must be relative
paths within `--audio`; duplicate files and path/symlink escapes are rejected.
Blank lines are ignored. Only manifest entries are evaluated, in their listed
order; unrelated files in the audio directory are ignored. Any format readable
by `AVAudioFile` can be supplied, such as AIFF, WAV, CAF, or M4A. Missing, empty,
or undecodable audio fails that clip and the remaining clips continue.

`--timeout <seconds>` bounds each clip's analysis (default 120; increase for long
recordings). Asset setup is a separate OS service operation, outside that
timeout. `--help` lists options. Exit status is 0 only when every listed clip
succeeds; bad arguments, malformed manifests, asset failures, and failed clips
exit 1. After valid input loading, asset/clip failures still produce both reports.
Failed clips are excluded from aggregate metrics and coverage is always shown.

## Metrics and interpretation

- Normalize with canonical Unicode composition, en-US POSIX lowercase, removal
  of Unicode punctuation/symbols, then Unicode whitespace splitting. Thus
  `don't` becomes `dont`, `twenty-one` becomes `twentyone`, and `21` stays `21`.
  No number expansion, accent folding, or filler removal. Punctuation without
  surrounding spaces joins its neighboring text.
- WER = `(substitutions + deletions + insertions) / reference word count`, using
  unit-cost word Levenshtein. Tied alignments prefer substitution, then deletion,
  then insertion; tied counts may differ from other aligners but WER is minimal.
  WER can exceed 100%. An empty reference has undefined clip WER; its insertion
  count still contributes to the corpus numerator. An all-empty reference corpus
  has undefined WER. Undefined/unavailable numeric fields are omitted from JSON
  and displayed as N/A in Markdown.
- Time-to-final is monotonic wall time from per-file setup through analyzer
  finalization and draining the result stream. It includes module creation,
  audio opening, decoding/conversion, lazy model loading, and analysis; excludes
  asset setup/download, WER computation, report writing, and Swift compilation.
  This is completion time for an unpaced file, not end-of-speech/live-release
  latency. Final segments are ordered by audio start time before joining.
- Real-time factor (RTF) = time-to-final / decoded audio duration; below 1 means
  faster than audio duration. Corpus WER and RTF use summed numerators and
  denominators, not averages of clip ratios. Mean time-to-final is also reported.
- Clips run sequentially with new analyzers and default model retention. No
  explicit warmup is performed; system caches can make later clips faster.
  Compare release builds on the same machine and corpus. Mac timing does not
  establish the iPhone 14 Pro capture or release-to-title latency budget.

`results.json` includes run metadata, source labels, setup timing/status, per-file
references and final hypotheses, edit counts, final-result counts, metrics,
errors, and coverage. `wer.md` is the readable version. No real recordings are
included yet. See [fixtures/synthetic/README.md](fixtures/synthetic/README.md).

The file-analysis/finalization lifecycle follows Apple's
[SpeechAnalyzer documentation](https://developer.apple.com/documentation/speech/speechanalyzer)
and the installed macOS SDK interfaces.
