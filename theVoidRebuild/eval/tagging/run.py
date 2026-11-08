"""Constrained MLX tagging benchmark. Every candidate gets a fresh worker process.

uv run --project eval python eval/tagging/run.py --limit 10
uv run --project eval python eval/tagging/run.py --data eval/data/sample.jsonl --output-dir eval/tagging/smoke
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SCHEMA, TAGS, TAXONOMY, MODEL_CACHE, MODEL_PATTERNS, configure_caches, digest, output_path, read_jsonl, write_json
configure_caches()


def load_rows(path, limit, split):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if row.get("label") not in TAGS or not isinstance(row.get("transcript"), str) or not row["transcript"].strip():
            raise ValueError(f"Invalid input row: {row.get('id')}")
        if not row.get("id") or row["id"] in seen:
            raise ValueError(f"Missing or duplicate input id: {row.get('id')}")
        seen.add(row["id"])
    if split != "all":
        rows = [r for r in rows if r.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No input rows selected")
    return rows


def environment():
    import mlx.core as mx
    import psutil
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor()
    return {"python": sys.version, "platform": platform.platform(), "macos": platform.mac_ver()[0],
            "chip": chip, "ram_gb": psutil.virtual_memory().total / 1e9,
            "mlx_device": str(mx.default_device()), "mlx_device_info": dict(mx.device_info()),
            "packages": {name: importlib.metadata.version(name) for name in
                         ["mlx", "mlx-lm", "lm-format-enforcer", "transformers", "huggingface-hub", "jsonschema"]}}


def infer(model, tokenizer, data, row, prompt_template, max_tokens):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from tagging.constrained import SchemaLogitsProcessor
    from tagging.metrics import validate_output

    result = {key: row.get(key) for key in ["id", "label", "split", "judge_label", "mixed", "quality_flags"]}
    result["split"] = row.get("split") or "unassigned"
    result.update(raw="", prediction=None, schema_valid=False, error=None, generation_tokens=0)
    mx.synchronize()
    mx.reset_peak_memory()
    start = time.perf_counter()
    try:
        prompt = prompt_template.replace("{transcript}", json.dumps(row["transcript"], ensure_ascii=False))
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False)
        if hasattr(tokens, "keys"):
            tokens = tokens["input_ids"]
        processor = SchemaLogitsProcessor(data, SCHEMA)
        ttft, response = None, None
        for response in stream_generate(model, tokenizer, tokens, max_tokens=max_tokens,
                                        sampler=make_sampler(temp=0), logits_processors=[processor]):
            if ttft is None:
                ttft = time.perf_counter() - start
            result["raw"] += response.text
        mx.synchronize()
        end = time.perf_counter()
        prediction, errors = validate_output(result["raw"])
        if response is None or processor.calls == 0:
            raise RuntimeError("Generation did not invoke schema constraints")
        result.update(prediction=prediction, schema_valid=not errors, schema_errors=errors,
                      finish_reason=response.finish_reason, prompt_tokens=response.prompt_tokens,
                      generation_tokens=response.generation_tokens, generation_tps=response.generation_tps,
                      latency_s=end - start, ttft_s=ttft, mlx_peak_gb=mx.get_peak_memory() / 1e9,
                      constraint_calls=processor.calls, constraint_cpu_s=processor.cpu_seconds,
                      chat_prompt_sha256=digest(tokens))
    except Exception as exc:
        mx.synchronize()
        result.update(error=f"{type(exc).__name__}: {exc}", latency_s=time.perf_counter() - start,
                      mlx_peak_gb=mx.get_peak_memory() / 1e9)
    return result


def worker(config_path):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from tagging.constrained import tokenizer_data
    from tagging.metrics import summarize

    config = json.loads(Path(config_path).read_text())
    candidate = config["candidate"]
    rows = config["rows"]
    output = Path(config["output"])
    payload = {"candidate": candidate, "rows": [], "status": "running", "environment": environment()}
    try:
        started = time.perf_counter()
        snapshot = snapshot_download(candidate["repo"], revision=candidate.get("revision", "main"),
                                     cache_dir=MODEL_CACHE, allow_patterns=MODEL_PATTERNS,
                                     local_files_only=config.get("offline", False))
        payload["download_or_cache_s"] = time.perf_counter() - started
        payload["resolved_revision"] = Path(snapshot).name
        started = time.perf_counter()
        model, tokenizer = load(snapshot)
        mx.eval(model.parameters())
        mx.synchronize()
        payload["load_s"] = time.perf_counter() - started
        payload["model_config"] = json.loads((Path(snapshot) / "config.json").read_text())
        bits = payload["model_config"].get("quantization", payload["model_config"].get("quantization_config", {})).get("bits")
        if bits != candidate["bits"]:
            raise ValueError(f"Expected {candidate['bits']}-bit model; config reports {bits}")
        started = time.perf_counter()
        data = tokenizer_data(tokenizer)
        payload["tokenizer_setup_s"] = time.perf_counter() - started
        payload["tokenizer_vocab_size"] = data.vocab_size
        prompt = config["prompt"]
        # Fixed separate memo: no selected test example gets a special extra prefill.
        warmup = {"id": "warmup", "label": "low_pleasant", "transcript": "I'm home now just sitting with my tea and feeling settled"}
        payload["warmups"] = [infer(model, tokenizer, data, warmup, prompt, config["max_tokens"])
                              for _ in range(config["warmup_n"])]
        if any(row.get("error") or not row["schema_valid"] for row in payload["warmups"]):
            raise RuntimeError("Constrained warmup failed; inspect warmups in worker output")
        for index, row in enumerate(rows):
            prediction = infer(model, tokenizer, data, row, prompt, config["max_tokens"])
            payload["rows"].append(prediction)
            if (index + 1) % 25 == 0 or index == len(rows) - 1:
                print(f"  {candidate['id']}: {index + 1}/{len(rows)}, valid={prediction['schema_valid']}", flush=True)
                write_json(output, payload)
        payload["metrics"] = summarize(payload["rows"])
        payload["status"] = "complete"
    except Exception as exc:
        payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(payload["error"], file=sys.stderr, flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload["process_peak_rss_gb"] = peak / (1e9 if sys.platform == "darwin" else 1e6)
    write_json(output, payload)
    return 0 if payload["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--candidates", type=Path, default=ROOT / "tagging" / "candidates.yaml")
    parser.add_argument("--models", nargs="+", help="Candidate IDs from YAML")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["all", "clean", "hard", "sample"], default="all")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--offline", action="store_true", help="Use cached model snapshots only, with all HF network access disabled")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tagging")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.worker:
        import mlx.core as mx
        config = json.loads(args.worker.read_text())
        mx.random.seed(config["seed"])
        return worker(args.worker)
    if args.limit is not None and args.limit < 1 or args.warmup < 1 or args.max_tokens < 1:
        parser.error("limit, warmup and max-tokens must be positive")
    import yaml
    from tagging.report import write_report
    dataset = args.data or (ROOT / "data" / "memos.jsonl")
    if not dataset.exists() and args.data is None:
        dataset = ROOT / "data" / "sample.jsonl"
        print("No generated dataset found; using the 10 handwritten sample rows", flush=True)
    rows = load_rows(dataset, args.limit, args.split)
    candidates = yaml.safe_load(args.candidates.read_text())["candidates"]
    if args.models:
        unknown = set(args.models) - {c["id"] for c in candidates}
        if unknown:
            parser.error(f"Unknown candidate IDs: {unknown}")
        candidates = [c for c in candidates if c["id"] in args.models]
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        parser.error("Candidate list must be nonempty with unique IDs")
    for candidate in candidates:
        if not candidate["id"].replace("-", "").replace(".", "").isalnum():
            parser.error("Candidate IDs must be simple filenames")
    output_dir = output_path(args.output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    prompt = (ROOT / "tagging" / "prompt.txt").read_text()
    if prompt.count("{transcript}") != 1:
        raise ValueError("Shared prompt requires exactly one transcript placeholder")
    payload = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "running", "expected_candidates": [c["id"] for c in candidates],
               "command": sys.argv, "dataset": str(dataset.resolve()), "selected_n": len(rows),
               "dataset_file_sha256": digest(dataset.read_bytes()), "selected_rows_sha256": digest(rows),
               "prompt_sha256": digest(prompt.encode()), "prompt": prompt, "schema": SCHEMA,
               "taxonomy": TAXONOMY, "seed": args.seed, "temperature": 0, "max_tokens": args.max_tokens,
               "warmup_n": args.warmup, "split_filter": args.split, "models": [],
               "offline_models": args.offline,
               "method": "All candidates: mlx-lm + LM Format Enforcer JSON Schema logits mask; native chat templates; enable_thinking=False; fresh process per candidate; sequential rows; no shared prompt/KV cache; invalid outputs count as errors."}
    manifest = dataset.with_suffix(".manifest.json")
    if manifest.exists():
        payload["data_manifest"] = json.loads(manifest.read_text())
    for candidate in candidates:
        print(f"Evaluating {candidate['id']} ({len(rows)} memos)", flush=True)
        result_path = run_dir / (candidate["id"] + ".json")
        config = dict(candidate=candidate, rows=rows, output=str(result_path), prompt=prompt,
                      max_tokens=args.max_tokens, warmup_n=args.warmup, seed=args.seed, offline=args.offline)
        config_path = run_dir / (candidate["id"] + ".config.json")
        write_json(config_path, config)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)])
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if completed.returncode or result["status"] != "complete":
                result["status"] = "failed"
                result.setdefault("error", f"Worker exited {completed.returncode}")
        else:
            result = {"candidate": candidate, "status": "failed", "error": f"Worker exited {completed.returncode}", "rows": []}
        payload["models"].append(result)
        if len(payload["models"]) == len(candidates):
            payload["status"] = "complete" if all(m["status"] == "complete" for m in payload["models"]) else "failed"
        write_json(run_dir / "results.json", payload)
        write_json(output_dir / "results.json", payload)
        write_report(payload, output_dir / "report.md", rows)
        write_report(payload, run_dir / "report.md", rows)
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    return 0 if all(m["status"] == "complete" for m in payload["models"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
