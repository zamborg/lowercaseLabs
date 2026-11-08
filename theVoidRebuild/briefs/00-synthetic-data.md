# Brief 00: synthetic labeled memos (Codex, using Haiku + a GPT model)

Write `eval/data/generate.py` that produces `eval/data/memos.jsonl` (lines: {id, transcript, label, split, gen_model, judge_model, judge_label}).

Pipeline:
1. GENERATE (model A): for each quadrant in `taxonomy.json`, make >= 75 first-person spoken voice-memo transcripts (target 300 total) from varied prompts: topic (work, family, health, money, creative, random errands), length (5 words to 150), speaker voice. Must read like speech to text: fillers (um, like), false starts, run-ons, no tidy punctuation, occasional ASR slips. Include ~20% mixed or near-neutral memos where a "dominant" quadrant is the target. Do NOT mention the label in the text.
2. JUDGE (model B, different vendor from A): given only the transcript and the taxonomy, return the quadrant via the same JSON schema. Blind to the target.
3. SPLIT: judge == target -> split "clean"; disagree -> split "hard" (kept, never dropped).
4. Dedupe near-duplicates (normalized trigram Jaccard > 0.8).
5. Fixed seeds; cache all model calls on disk so reruns are free; print counts per quadrant per split.

Also ship `eval/data/sample.jsonl` (10 hand-written rows) so the eval runs without API calls.
Done = generate.py --n 300 yields memos.jsonl, balanced within 10% per quadrant in "clean", report of agreement rate.
