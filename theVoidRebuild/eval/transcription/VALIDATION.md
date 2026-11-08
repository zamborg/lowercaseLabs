# Validation — November 7, 2026 (Pacific)

Host: macOS 26.5.2 (25F84), arm64; Apple Swift 6.3.3; macOS 26.5 SDK.
All implementation, fixtures, reports, and temporary verification files are
under `eval/transcription/`. No real recordings were available.

Commands were run from this directory:

```sh
swift test
swift test -c release
python3 scripts/make_synthetic.py
swift run wer-eval --audio fixtures/synthetic/audio --ref fixtures/synthetic/reference.jsonl
swift run -c release wer-eval --audio fixtures/synthetic/audio --ref fixtures/synthetic/reference.jsonl
```

Both test configurations passed **13 Swift Testing tests**, covering WER
normalization, Unicode, all edit types, mixed edits, repeated words, tied
alignments, empty inputs, WER above 100%, weighted corpus scoring, minimal
manifests, synthetic provenance, malformed/duplicate paths, and symlink escapes.
The first test attempt contained an incorrect repeated-word expectation; that
expectation was corrected and both configurations subsequently passed.

Fixture generation passed with the installed Samantha voice. `/usr/bin/afinfo`
confirmed nonzero audio data and durations of 3.606395, 6.078277, and 7.077098
seconds. The generator now rejects header-only output even when `say` exits 0.

Both the default and release CLI commands succeeded on all three synthetic
clips. The final checked-in [wer.md](wer.md) and [results.json](results.json)
come from the final release run:

| Metric | Result |
| --- | ---: |
| Clips succeeded | 3 / 3 |
| Reference words | 55 |
| Substitutions / deletions / insertions | 1 / 0 / 0 |
| Corpus WER | 1.82% |
| Audio duration | 16.762 s |
| Total time-to-final | 0.550 s |
| Mean time-to-final | 0.183 s |
| Corpus real-time factor | 0.033 |

The one substitution is reference `first` versus hypothesis `1st`. Number
expansion is intentionally absent from normalization. These are repeated runs
on synthetic audio with possible warm system caches; they are not cold-start
or representative human-recording benchmarks.

Additional manual CLI checks used temporary manifests under the ignored
`.build/verification/` directory:

```sh
swift run wer-eval --audio fixtures/synthetic/audio --ref .build/verification/mixed.jsonl --output .build/verification/mixed
swift run wer-eval --audio fixtures/synthetic/audio --ref .build/verification/one.jsonl --timeout 0.000001 --output .build/verification/timeout
.build/debug/wer-eval --help
.build/debug/wer-eval --audio fixtures/synthetic/audio --ref fixtures/synthetic/reference.jsonl --timeout nan
```

- `mixed.jsonl` listed a nonexistent `missing.wav` followed by the first
  synthetic clip. Exit 1; both reports written; the valid clip still completed;
  1/2 coverage and only its 12 reference words contributed to aggregates.
- `one.jsonl` listed the first synthetic clip. A one-microsecond timeout exited
  1 without hanging; reports marked the clip failed, WER unavailable, and no
  successful timing or score was fabricated.
- Help exited 0. Invalid nonfinite timeout exited 1 before speech setup.
- Python assertions checked JSON coverage, failure exclusion, absent metrics on
  failure, final-result counts, source labels, edit totals, and the summed RTF
  arithmetic against the final release report. All passed.

## Availability, permissions, and limits

The initial speech-service check reported en-US assets as `supported`, not
`installed`. The system installation-request path completed successfully, and
actual finalized transcriptions were collected. A fresh executable invocation
could again report `supported`; final setup therefore always obtains the
system's installation request and ensures readiness by default. Asset setup
is excluded from clip timing. The tool releases only a locale reservation it
acquired itself.

The execution sandbox initially blocked SwiftPM's standard compiler cache and
caused `say` to emit header-only audio despite exit 0. Tests, fixture generation,
and speech evaluation succeeded with approved execution outside that sandbox.
No microphone access or legacy `SFSpeechRecognizer` authorization was requested.
This does not verify behavior under other sandbox, MDM, file-access, or permission
restrictions. Offline model setup and unsupported hardware were not tested.

Human-recording WER, noisy audio, long recordings, microphone capture, live
end-of-speech latency, and iPhone 14 Pro performance remain unverified. The owner
must supply the planned recordings and corrected references for that evaluation.
