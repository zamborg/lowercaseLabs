from __future__ import annotations

from collections import Counter
from pathlib import Path

from common import TAGS, output_path


def pct(value):
    return f"{100 * value:.1f}%" if value is not None else "—"


def number(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "—"


def ranking_score(model):
    metrics = model["metrics"]
    primary = metrics["by_split"].get("clean", metrics["all"])
    return (-(primary["macro_top1"] or 0), -(primary["top1"] or 0),
            -metrics["schema_valid_rate"], metrics["latency_s"]["median"] or float("inf"))


def write_report(payload, destination, dataset_rows):
    models = sorted([model for model in payload["models"] if model["status"] == "complete"], key=ranking_score)
    rows_by_id = {row["id"]: row for row in dataset_rows}
    lines = ["# The Void tagging evaluation", "", f"Run `{payload['run_id']}` · {payload['selected_n']} memos · temperature 0", "",
             "Synthetic screening results on this Mac. These are not real-memo accuracy or iPhone latency measurements.", "",
             "## Ranking", "",
             "Ranked by clean-split macro top-1, then clean top-1, schema validity, and median full-output latency. When no clean split exists, uses all selected rows. No prompt tuning on these results.", "",
             "| Rank | Model (4-bit) | Clean top-1 | All top-1 | Any tag | Schema valid | Median / p95 (s) | Decode tok/s | End-to-end tok/s | MLX peak GB | RSS peak GB |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if payload.get("status") == "running":
        lines[2] += f" · IN PROGRESS ({len(payload['models'])}/{len(payload['expected_candidates'])} candidates finished)"
    for rank, model in enumerate(models, 1):
        m = model["metrics"]
        clean = m["by_split"].get("clean", {})
        lines.append(f"| {rank} | {model['candidate']['id']} | {pct(clean.get('top1'))} | {pct(m['all']['top1'])} | {pct(m['all']['any_tag'])} | {pct(m['schema_valid_rate'])} | {number(m['latency_s']['median'])} / {number(m['latency_s']['p95'])} | {number(m['generation_tps_median'], 1)} | {number(m['end_to_end_tps'], 1)} | {number(m['mlx_peak_gb'])} | {number(model['process_peak_rss_gb'])} |")
    failed = [model for model in payload["models"] if model["status"] != "complete"]
    for model in failed:
        lines += ["", f"**FAILED: {model['candidate']['id']}** — {model.get('error', 'worker failed')}. Not ranked; see worker JSON."]
    if models:
        leader = models[0]
        lines += ["", f"**Screening leader: `{leader['candidate']['id']}`.** Use this as a shortlist for real holdout and iPhone testing. The ranking rule prioritizes tag accuracy; speed and memory remain visible rather than being combined into an arbitrary score."]
        primary = leader["metrics"]["by_split"].get("clean", leader["metrics"]["all"])
        if primary["top1"] < .7:
            lines += ["", "**Dominant-tag accuracy is weak in this baseline.** Calibrate a prompt on a separate development set and validate a human-labeled real holdout before adopting a production tagger. The small differences at the top of this table do not establish a statistically reliable winner."]
    lines += ["", "## Dataset and label reliability", "",
              f"Source: `{Path(payload['dataset']).name}`, SHA-256 `{payload['dataset_file_sha256']}`.",
              f"Selected rows: `{payload['selected_rows_sha256']}`. Splits: `{dict(Counter(r.get('split', 'unassigned') for r in dataset_rows))}`.", ""]
    manifest = payload.get("data_manifest")
    if manifest:
        lines += [f"Generator `{manifest['gen_model']}`; blind judge `{manifest['judge_model']}`. Dataset-wide agreement: **{pct(manifest['agreement_rate'])}** over {manifest['retained_n']} retained memos. Mixed/near-neutral requested: {manifest['mixed_n']}. Normalized word-trigram Jaccard >0.8 (or exact normalized match) removes near-duplicates before judging; removed {len(manifest['duplicates_removed'])}.", "",
                  "| Generator target | Clean | Hard |", "|---|---:|---:|"]
        for tag in TAGS:
            lines.append(f"| {tag} | {manifest['counts']['clean'][tag]} | {manifest['counts']['hard'][tag]} |")
        lines += ["", f"Clean balance min/max: {pct(manifest['clean_balance_min_over_max'])}. Actual transcript lengths: {manifest['word_count_min']}–{manifest['word_count_max']} whitespace-delimited words; {manifest['requested_length_range_misses']} miss their narrower requested length bucket. **{manifest.get('long_letter_run_n', 0)} memos contain 25+ consecutive letters**, flagging possible joined-word artifacts caused by the generation word-count regex. They are retained; the clean-without-artifacts sensitivity score below checks their effect. Zero bucket misses does not imply fluent transcripts.", "",
                  "The judge sees only taxonomy, a fixed rubric and transcript: no target label, generation metadata or model predictions. Separate model snapshots provide a blind consistency check, but both are OpenAI models. The independence loss cannot be quantified from this dataset: shared training, alignment and emotion conventions can create correlated mistakes. Agreement is not human-verified accuracy. A second vendor would reduce one source of correlation, not establish ground truth.", "",
                  "Clean labels are model agreements and likely easier than real memos. Hard labels remain generator targets with unresolved judge disagreement; all-vs-target and hard-vs-judge scores show sensitivity. Disagreements are never relabeled or deleted. Any supplementary generation only balances clean counts and is disclosed in the manifest."]
    else:
        lines += ["Handwritten smoke-test data; no automated generator/judge agreement is asserted."]
    lines += ["", "## Accuracy per quadrant and split", "",
              "Top-1 is the first tag; any-tag accepts either returned tag. Invalid outputs count as wrong. Per-quadrant accuracy is recall on that target quadrant. A second tag can inflate any-tag accuracy; its frequency is reported below."]
    for model in models:
        m = model["metrics"]
        lines += ["", f"### {model['candidate']['id']}", "",
                  "| Split | N | Top-1 | Any tag | Macro top-1 | Top-1 Wilson 95% interval |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, a in [("all vs target", m["all"]), *m["by_split"].items(), ("hard vs judge", m["hard_vs_judge"]),
                        ("clean without long letter runs", m["clean_without_long_letter_runs"]),
                        ("flagged long letter runs", m["flagged_long_letter_runs"]),
                        ("mixed vs target", m["mixed"]), ("non-mixed vs target", m["not_mixed"])]:
            ci = a.get("top1_wilson95")
            lines.append(f"| {name} | {a['n']} | {pct(a['top1'])} | {pct(a['any_tag'])} | {pct(a['macro_top1'])} | {pct(ci[0]) + '–' + pct(ci[1]) if ci else '—'} |")
        lines += ["", "| Target quadrant | All N | All top-1 | All any-tag | Clean top-1 | Hard top-1 |", "|---|---:|---:|---:|---:|---:|"]
        for tag in TAGS:
            a = m["all"]["by_quadrant"][tag]
            clean = m["by_split"].get("clean", {}).get("by_quadrant", {}).get(tag, {})
            hard = m["by_split"].get("hard", {}).get("by_quadrant", {}).get(tag, {})
            lines.append(f"| {tag} | {a['n']} | {pct(a['top1'])} | {pct(a['any_tag'])} | {pct(clean.get('top1'))} | {pct(hard.get('top1'))} |")
        lines += ["", "Confusion matrix, all rows: true quadrant in rows, first predicted tag in columns.", "",
                  "| True \\ predicted | " + " | ".join(TAGS + ["invalid"]) + " |", "|---|" + "---:|" * 5]
        for tag in TAGS:
            lines.append("| " + tag + " | " + " | ".join(str(m["all"]["confusion"][tag][p]) for p in TAGS + ["invalid"]) + " |")
        lines += ["", f"Second-tag rate: {pct(m['two_tags_rate'])}; duplicate-tag outputs: {m['duplicate_tags_n']}; blank titles: {m['blank_titles_n']}; median title length: {number(m['title_chars_median'], 1)} characters. Completed {m['completed_n']}/{m['n']}; truncated {m['truncated_n']}; inference errors {m['errors_n']}.", "",
                  f"Model revision `{model['resolved_revision']}`. Model load {number(model['load_s'])} s; tokenizer trie setup {number(model['tokenizer_setup_s'])} s; download/cache lookup {number(model['download_or_cache_s'])} s. These are excluded from warm latency. Median first yielded token {number(m['ttft_s_median'])} s."]
    lines += ["", "## Titles and error inspection", "",
              "Title syntax/length is measured. Semantic title quality has no reference labels or validated score; examples below are inspection material, not a title-quality ranking. Raw outputs for every row are in results.json.", "",
              "| Model | Memo | Transcript (excerpt) | Title | First tag / target |", "|---|---|---|---|---|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for model in models:
        predictions = model["rows"]
        # One fixed example and two errors (if present), deterministic and disclosed.
        selected = predictions[:1] + [r for r in predictions[1:] if not r["schema_valid"] or r["prediction"]["tags"][0] != r["label"]][:2]
        if len(selected) < 3:
            selected += [r for r in predictions[1:] if r not in selected][:3 - len(selected)]
        for row in selected:
            prediction = row.get("prediction") or {}
            transcript = rows_by_id[row["id"]]["transcript"]
            excerpt = transcript[:180] + ("…" if len(transcript) > 180 else "")
            first = (prediction.get("tags") or ["invalid"])[0]
            lines.append(f"| {model['candidate']['id']} | {row['id']} | {cell(excerpt)} | {cell(prediction.get('title', row.get('error')))} | {first} / {row['label']} |")
    lines += ["", "## Measurement and reproduction", "",
              payload["method"], "",
              f"{payload['warmup_n']} constrained warmups per model on a separate fixed memo. Greedy temperature 0, seed {payload['seed']}, max {payload['max_tokens']} generated tokens. A fresh schema parser per memo, shared tokenizer trie per model. Canonical JSON field order and at most two consecutive formatting whitespaces. No unconstrained fallback or output repair; the raw response is checked independently with jsonschema, including title ≤60 characters, exact enum values and 1–2 tags.", "",
              "Full-output latency includes prompt rendering/tokenization, prefill, decode, Python schema masking and GPU synchronization. It excludes model download/load, tokenizer-trie construction, warmups and result I/O. p95 uses nearest rank. Decode tok/s is the median mlx-lm generation_tps (includes schema-mask overhead, excludes prefill; counts its terminal token). End-to-end tok/s is total generation_tokens / measured full-output seconds. Different tokenizers make tok/s less comparable than full latency.", "",
              "MLX peak is the maximum per-request MLX allocator peak, reset before each measured request, including resident weights and KV buffers; it is not total process memory. RSS peak is OS process high-water memory for the isolated worker, including loading and trie setup. Do not add these unified-memory figures together. Latency percentiles cover completed outputs; errors and truncations are reported separately and still count as incorrect in accuracy.", "",
              "Wilson intervals describe finite synthetic-example sampling only; they do not cover label uncertainty, generation bias, paired model differences or hardware variability. One sequential timing pass is not a production latency guarantee. Results are specific to this fixed zero-shot prompt; no few-shot prompting, fine-tuning or prompt-cache optimization was evaluated.", ""]
    if models:
        env = models[0]["environment"]
        lines += [f"Host: **{env['chip']}**, {number(env['ram_gb'], 1)} GB RAM, macOS {env['macos']}, {env['mlx_device']}; Python {env['python'].split()[0]}.", "",
                  "Dependencies: " + ", ".join(f"`{k}=={v}`" for k, v in env["packages"].items()) + ". Full dependency graph is pinned in eval/uv.lock.", ""]
    lines += [f"Prompt SHA-256 `{payload['prompt_sha256']}`. Schema, exact model commits, inputs, environment, warmups and row-level timing are saved with each timestamped run.", "",
              "```sh", "# From the repository root", "export UV_CACHE_DIR=\"$PWD/eval/.cache/uv\"", "uv sync --project eval --frozen",
              "uv run --project eval python eval/data/generate.py --n 300 --offline",
              "uv run --project eval python eval/tagging/run.py --limit 10",
              "uv run --project eval python eval/tagging/run.py",
              "uv run --project eval python eval/tagging/run.py --data eval/data/sample.jsonl --output-dir eval/tagging/smoke", "```", "",
              "## Deviations and remaining checks", "",
              "- The MLX/API pipeline uses Python 3.13 via uv, rather than system Python 3.9. An eval-local pyproject keeps all changes under eval/, so commands use `--project eval`.",
              "- Different dated OpenAI models replace the unavailable Anthropic generator/judge pairing. The judge is blind, but vendor independence is absent and the resulting confidence loss is unquantified.",
              "- Four 1.2–1.7B candidates include Liquid LFM2; Qwen2.5 1.5B adds another ungated size-matched instruct baseline instead of a >2B model.",
              "- An initial 12-memo-per-request pilot had only 60.7% generator/judge agreement and 233/387 length-bucket misses. It is preserved under `eval/data/pilot/`, including every disagreement, and is excluded from the main ranking. The final generator uses one memo per request with schema-constrained word counts to avoid batch alignment drift. Word constraints introduced joined-word artifacts; conservative quality flags and a sensitivity score expose them. All pilot and final API calls remain cached. Future data generation should prefer length repair over a hard word-count regex.",
              "- Hard labels are unresolved. No human-labeled real holdout, title-fidelity benchmark, real ASR-error distribution, multilingual coverage or iPhone 14 Pro / MLX Swift constrained-decoder timing has been verified.",
              "- The ~2-second app target must be tested on the iPhone, including transcript-finalization and capture-to-save costs; this Mac benchmark excludes those stages.", "",
              "Implementation references: [mlx-lm](https://github.com/ml-explore/mlx-lm), [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).", ""]
    output_path(destination).write_text("\n".join(lines))
