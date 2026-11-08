"""Revalidate saved outputs/metrics and regenerate a report without loading models.

Optionally compare a repeat run's raw outputs for deterministic reproduction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SCHEMA, digest, output_path, read_jsonl, write_json
from tagging.metrics import summarize, validate_output
from tagging.report import write_report


def audit(path, compare=None):
    path = output_path(path)
    payload = json.loads(path.read_text())
    if payload["schema"] != SCHEMA:
        raise ValueError("Saved schema differs from the current schema")
    if digest(payload["prompt"].encode()) != payload["prompt_sha256"]:
        raise ValueError("Saved prompt hash mismatch")
    dataset = Path(payload["dataset"])
    if digest(dataset.read_bytes()) != payload["dataset_file_sha256"]:
        raise ValueError("Dataset has changed since this run")
    originals = {row["id"]: row for row in read_jsonl(dataset)}
    repeated = {}
    if compare:
        repeat_payload = json.loads(Path(compare).read_text())
        if repeat_payload["prompt_sha256"] != payload["prompt_sha256"]:
            raise ValueError("Repeat run used a different prompt")
        repeated = {model["candidate"]["id"]: model for model in repeat_payload["models"]}
    checked, repeated_n, summaries = 0, 0, []
    for model in payload["models"]:
        name = model["candidate"]["id"]
        if model["status"] != "complete" or len(model["rows"]) != payload["selected_n"]:
            raise ValueError(f"{name}: incomplete model run")
        selected = [originals[row["id"]] for row in model["rows"]]
        if digest(selected) != payload["selected_rows_sha256"]:
            raise ValueError(f"{name}: selected rows differ from saved hash")
        for row in model["rows"]:
            parsed, errors = validate_output(row["raw"])
            if row["schema_valid"] != (not errors) or parsed != row["prediction"]:
                raise ValueError(f"{name}/{row['id']}: raw schema validity/prediction mismatch")
            if not row.get("error") and row.get("constraint_calls", 0) == 0:
                raise ValueError(f"{name}/{row['id']}: no schema processor calls")
            checked += 1
        recalculated = summarize(model["rows"])
        if recalculated != model["metrics"]:
            raise ValueError(f"{name}: saved metrics differ from recomputed metrics")
        matched = 0
        if name in repeated:
            repeat_model = repeated[name]
            if repeat_model["status"] != "complete" or repeat_model["resolved_revision"] != model["resolved_revision"]:
                raise ValueError(f"{name}: repeat run failed or used a different model revision")
            by_id = {row["id"]: row for row in model["rows"]}
            for row in repeat_model["rows"]:
                if row["id"] not in by_id or row["raw"] != by_id[row["id"]]["raw"]:
                    raise ValueError(f"{name}/{row['id']}: repeat output differs")
                matched += 1
            repeated_n += matched
        summaries.append({"model": name, "rows": len(model["rows"]),
                          "schema_valid": sum(row["schema_valid"] for row in model["rows"]),
                          "constraint_calls_min": min(row.get("constraint_calls", 0) for row in model["rows"]),
                          "repeat_rows_identical": matched,
                          "clean_top1": recalculated["by_split"].get("clean", {}).get("top1")})
    selected = [originals[row["id"]] for row in payload["models"][0]["rows"]]
    write_report(payload, path.parent / "report.md", selected)
    archived_report = path.parent / "runs" / payload["run_id"] / "report.md"
    if archived_report.parent.exists():
        write_report(payload, archived_report, selected)
    result = {"audited_at": datetime.now(timezone.utc).isoformat(), "run_id": payload["run_id"],
              "results_sha256": digest(path.read_bytes()), "rows_checked": checked,
              "repeat_rows_identical": repeated_n, "models": summaries,
              "checks": ["schema against raw outputs", "saved predictions", "constraint invocation",
                         "dataset/selected-input hashes", "prompt hash", "recomputed metrics", "repeat raw outputs when provided"]}
    write_json(path.with_suffix(".audit.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "tagging" / "results.json")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    audit(args.results, args.compare)
