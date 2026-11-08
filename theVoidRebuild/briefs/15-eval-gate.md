# Brief 15: ML eval gate and the improvement loop

**Executor:** Codex, via `~/.shepherd`, as its own long-running workstream. **Depends on:** nothing. It can start now, in parallel with briefs 10–14.

**Touches only:** `eval/`, `app/MacHarness/main.swift` (the `--eval` section), `app/Sources/Tagger.swift` (the prompt and the model directory, nothing else), and `app/tagging-baseline.txt`. Other agents may be editing the rest of `app/` at the same time.

## Goal

Make tagging and titling quality improvable without Zubin, behind a gate that stops regressions.

## Part 1: the gate

- `app/validate.sh --eval` already has code, written but never run, that tags the clean split of `eval/data/memos.jsonl` with the **shipped** `Tagger.swift` and compares top-1 accuracy to `app/tagging-baseline.txt`. Run it, fix it, and store the first baseline (`--update-baseline`).
- Add the hard split as a second number. Report it, but don't gate on it.
- Add per-quadrant accuracy and the confusion pairs to the output.
- Runtime target: under 5 minutes on this Mac.

## Part 2: title quality

There's no metric today, and titles are weak. Examples from the phone: "Energetic Excitement", "Emotional State – Low Pleasant". Build a cheap automated check:

- Hard rules that fail: the title contains a quadrant name or id, uses the words "emotional state", is over 60 characters, or repeats the transcript's first words verbatim.
- A judged score using a model *different* from whatever generated the data, comparing against the transcript ("is this a specific, faithful title a person would write?"). Run it on demand, not in the gate.

## Part 3: improve

Iterate on the prompt, and on candidate models from `eval/tagging/candidates.yaml`, within the latency budget. **Phone latency matters:** LFM2-1.2B measured about 1.45 s per message on the iPhone 14 Pro (Release build). Anything bigger needs a phone measurement (rig D) before it can ship. Ask the coordinating agent to run it.

A change lands only per [AGENTS.md](../AGENTS.md#the-ml-loop): accuracy at or above baseline, rule violations down, latency within budget, baseline updated in the same commit.

## Report

`eval/tagging/report.md`, updated each round: accuracy (clean and hard), per-quadrant numbers, title rule violations, judged title score, latency, and a 20-row before/after title table.
