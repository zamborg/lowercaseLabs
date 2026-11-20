from __future__ import annotations

from collections import Counter
import json
import math
import statistics

from jsonschema import Draft202012Validator
from common import SCHEMA, TAGS

VALIDATOR = Draft202012Validator(SCHEMA)


def validate_output(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate object key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=unique_object)
        errors = sorted(VALIDATOR.iter_errors(parsed), key=lambda error: str(error.path))
        return parsed, [error.message for error in errors]
    except (ValueError, TypeError) as exc:
        return None, [str(exc)]


def percentile(values, fraction):
    """Nearest-rank quantile, including p95 on small smoke sets."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0, center - width), min(1, center + width)]


def accuracy(rows, label_key="label"):
    eligible = [row for row in rows if row.get(label_key) in TAGS]
    if not eligible:
        return {"n": 0, "top1": None, "any_tag": None, "macro_top1": None, "by_quadrant": {}, "confusion": {}}
    confusion = {tag: {pred: 0 for pred in TAGS + ["invalid"]} for tag in TAGS}
    top1 = any_tag = 0
    by_quadrant = {}
    for row in eligible:
        tags = row["prediction"]["tags"] if row["schema_valid"] else []
        predicted = tags[0] if tags else "invalid"
        confusion[row[label_key]][predicted] += 1
        top1 += predicted == row[label_key]
        any_tag += row[label_key] in tags
    for tag in TAGS:
        subset = [row for row in eligible if row[label_key] == tag]
        correct = confusion[tag][tag]
        by_quadrant[tag] = {"n": len(subset), "top1": correct / len(subset) if subset else None,
                            "any_tag": sum(row["schema_valid"] and tag in row["prediction"]["tags"] for row in subset) / len(subset) if subset else None}
    supported = [value["top1"] for value in by_quadrant.values() if value["n"]]
    return {"n": len(eligible), "top1": top1 / len(eligible), "any_tag": any_tag / len(eligible),
            "top1_wilson95": wilson(top1, len(eligible)),
            "macro_top1": statistics.mean(supported), "by_quadrant": by_quadrant, "confusion": confusion}


def summarize(rows):
    complete = [row for row in rows if not row.get("error") and row.get("finish_reason") == "stop"]
    latencies = [row["latency_s"] for row in complete]
    generated = [row for row in rows if row.get("generation_tokens")]
    valid = [row for row in rows if row["schema_valid"]]
    split_counts = Counter(row.get("split") or "unassigned" for row in rows)
    return {
        "n": len(rows), "schema_valid_rate": len(valid) / len(rows) if rows else None,
        "completed_n": len(complete), "errors_n": sum(bool(row.get("error")) for row in rows),
        "truncated_n": sum(row.get("finish_reason") == "length" for row in rows),
        "all": accuracy(rows),
        "by_split": {split: accuracy([r for r in rows if (r.get("split") or "unassigned") == split]) for split in sorted(split_counts)},
        "hard_vs_judge": accuracy([r for r in rows if r.get("split") == "hard"], "judge_label"),
        "clean_without_long_letter_runs": accuracy([r for r in rows if r.get("split") == "clean" and "long_letter_run" not in (r.get("quality_flags") or [])]),
        "flagged_long_letter_runs": accuracy([r for r in rows if "long_letter_run" in (r.get("quality_flags") or [])]),
        "mixed": accuracy([r for r in rows if r.get("mixed") is True]),
        "not_mixed": accuracy([r for r in rows if r.get("mixed") is False]),
        "latency_s": {"median": statistics.median(latencies) if latencies else None, "p95": percentile(latencies, .95),
                      "completed_n": len(latencies), "definition": "Wall time from prompt rendering through full constrained output and GPU synchronization; excludes load, tokenizer trie setup, warmup, checkpoint I/O"},
        "generation_tps_median": statistics.median(r["generation_tps"] for r in generated) if generated else None,
        "end_to_end_tps": sum(r.get("generation_tokens", 0) for r in generated) / sum(r["latency_s"] for r in generated) if generated else None,
        "ttft_s_median": statistics.median(r["ttft_s"] for r in generated if r.get("ttft_s") is not None) if any(r.get("ttft_s") is not None for r in generated) else None,
        "mlx_peak_gb": max((r.get("mlx_peak_gb", 0) for r in rows), default=0),
        "two_tags_rate": sum(len(r["prediction"]["tags"]) == 2 for r in valid) / len(valid) if valid else None,
        "duplicate_tags_n": sum(len(set(r["prediction"]["tags"])) != len(r["prediction"]["tags"]) for r in valid),
        "blank_titles_n": sum(not r["prediction"]["title"].strip() for r in valid),
        "title_chars_median": statistics.median(len(r["prediction"]["title"]) for r in valid) if valid else None,
    }
