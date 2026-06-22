"""Utilities for analyzing published WolfBench run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

WOLFBENCH_GITHUB_RUNS_URL = (
    "https://github.com/wandb/WolfBench/tree/main/wolfbench-runs"
)
_GITHUB_API_RUNS_PATH = (
    "https://api.github.com/repos/wandb/WolfBench/contents/wolfbench-runs"
)
_GITHUB_API_RUNS_URL = f"{_GITHUB_API_RUNS_PATH}?ref=main"
_RAW_RUNS_BASE_URL = (
    "https://raw.githubusercontent.com/wandb/WolfBench/main/wolfbench-runs"
)


@dataclass(frozen=True)
class _RunRef:
    config_name: str
    run_name: str
    result_uri: str | Path
    config_uri: str | Path

    @property
    def run_key(self) -> str:
        return f"{self.config_name}/{self.run_name}"


def build_wolfbench_report(
    source: str | Path,
    output_dir: str | Path,
    *,
    max_runs: int | None = None,
    min_runs: int = 2,
) -> dict[str, Any]:
    """Build CSV, plot, and HTML summaries from WolfBench run-level JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_df, task_run_df = load_wolfbench_run_data(source, max_runs=max_runs)
    config_summary = build_wolfbench_config_summary(run_df, task_run_df)
    task_summary = build_wolfbench_task_summary(task_run_df)
    task_config_summary = build_wolfbench_task_config_summary(task_run_df)
    consistency_contrast_summary = build_wolfbench_consistency_contrast_summary(
        config_summary,
        min_runs=min_runs,
    )

    run_df.to_csv(output_path / "run_metrics.csv", index=False)
    task_run_df.to_csv(output_path / "task_run_metrics.csv", index=False)
    config_summary.to_csv(output_path / "config_summary.csv", index=False)
    task_summary.to_csv(output_path / "task_summary.csv", index=False)
    task_config_summary.to_csv(output_path / "task_config_summary.csv", index=False)
    consistency_contrast_summary.to_csv(
        output_path / "consistency_contrast_summary.csv",
        index=False,
    )

    plot_paths = _write_plots(
        config_summary=config_summary,
        task_summary=task_summary,
        output_dir=output_path,
        min_runs=min_runs,
    )
    html_path = _write_html_report(
        source=str(source),
        output_dir=output_path,
        run_df=run_df,
        task_run_df=task_run_df,
        config_summary=config_summary,
        task_summary=task_summary,
        consistency_contrast_summary=consistency_contrast_summary,
        plot_paths=plot_paths,
        min_runs=min_runs,
    )

    return {
        "source": str(source),
        "runs": int(len(run_df)),
        "task_runs": int(len(task_run_df)),
        "configs": int(run_df["config_name"].nunique()) if not run_df.empty else 0,
        "tasks": int(task_run_df["task_name"].nunique()) if not task_run_df.empty else 0,
        "output_dir": str(output_path),
        "html_report": str(html_path),
        "plots": [str(path) for path in plot_paths],
    }


def load_wolfbench_run_data(
    source: str | Path,
    *,
    max_runs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WolfBench run-level and per-task reward rows from a local path or GitHub URL."""
    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for run_ref in _iter_run_refs(source, max_runs=max_runs):
        result = _load_json(run_ref.result_uri)
        config = _load_json(run_ref.config_uri, required=False) or {}
        next_run_rows, next_task_rows = _records_from_run(run_ref, result, config)
        run_rows.extend(next_run_rows)
        task_rows.extend(next_task_rows)

    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        run_df = pd.DataFrame(
            columns=[
                "run_key",
                "config_name",
                "run_name",
                "eval_key",
                "job_id",
                "agent",
                "model",
                "model_display",
                "thinking_display",
                "dataset",
                "started_at",
                "finished_at",
                "duration_seconds",
                "n_total_trials",
                "n_trials",
                "n_errors",
                "score_mean",
                "solved_tasks",
                "failed_tasks",
                "reward_task_count",
                "exception_count",
                "exception_types",
            ]
        )

    task_run_df = pd.DataFrame(task_rows)
    if task_run_df.empty:
        task_run_df = pd.DataFrame(
            columns=[
                "run_key",
                "config_name",
                "run_name",
                "eval_key",
                "agent",
                "model",
                "model_display",
                "thinking_display",
                "dataset",
                "task_name",
                "trial_name",
                "reward",
                "success",
                "exception_type",
            ]
        )

    return run_df, task_run_df


def build_wolfbench_config_summary(
    run_df: pd.DataFrame,
    task_run_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize WolfBench five-metric consistency by configuration."""
    if run_df.empty:
        return pd.DataFrame(
            columns=[
                "config_name",
                "agent",
                "model",
                "model_display",
                "thinking_display",
                "runs",
                "tasks",
                "average_score",
                "best_of_score",
                "worst_of_score",
                "ceiling_score",
                "solid_score",
                "score_spread",
                "ceiling_minus_solid",
            ]
        )

    rows: list[dict[str, Any]] = []
    for config_name, run_group in run_df.groupby("config_name", dropna=False):
        run_keys = run_group["run_key"].tolist()
        task_group = task_run_df[task_run_df["run_key"].isin(run_keys)].copy()
        pivot = _success_pivot(task_group)
        ceiling_score = float(pivot.any(axis=1).mean()) if not pivot.empty else None
        solid_score = float(pivot.all(axis=1).mean()) if not pivot.empty else None
        score_mean = pd.to_numeric(run_group["score_mean"], errors="coerce")
        duration = pd.to_numeric(run_group["duration_seconds"], errors="coerce")
        errors = pd.to_numeric(run_group["n_errors"], errors="coerce")
        trials = pd.to_numeric(run_group["n_trials"], errors="coerce")
        first = run_group.iloc[0]
        best = score_mean.max()
        worst = score_mean.min()
        rows.append(
            {
                "config_name": config_name,
                "agent": first.get("agent"),
                "model": first.get("model"),
                "model_display": first.get("model_display"),
                "thinking_display": first.get("thinking_display"),
                "runs": int(len(run_group)),
                "tasks": int(task_group["task_name"].nunique()) if not task_group.empty else 0,
                "average_score": float(score_mean.mean()) if score_mean.notna().any() else None,
                "best_of_score": float(best) if pd.notna(best) else None,
                "worst_of_score": float(worst) if pd.notna(worst) else None,
                "ceiling_score": ceiling_score,
                "solid_score": solid_score,
                "score_spread": (
                    float(best - worst) if pd.notna(best) and pd.notna(worst) else None
                ),
                "ceiling_minus_solid": (
                    ceiling_score - solid_score
                    if ceiling_score is not None and solid_score is not None
                    else None
                ),
                "mean_duration_minutes": (
                    float(duration.mean() / 60) if duration.notna().any() else None
                ),
                "mean_errors": float(errors.mean()) if errors.notna().any() else None,
                "mean_trials": float(trials.mean()) if trials.notna().any() else None,
                "total_exceptions": int(run_group["exception_count"].sum()),
            }
        )

    return _round_numeric(
        pd.DataFrame(rows).sort_values(
            ["average_score", "solid_score", "ceiling_score"],
            ascending=[False, False, False],
            na_position="last",
        )
    )


def build_wolfbench_task_summary(task_run_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize task-level difficulty across all WolfBench configurations."""
    if task_run_df.empty:
        return pd.DataFrame(
            columns=[
                "task_name",
                "runs",
                "configs",
                "success_rate",
                "solved_anywhere",
                "solved_everywhere",
                "exception_rate",
            ]
        )
    frame = task_run_df.copy()
    frame["exception_seen"] = frame["exception_type"].notna()
    summary = (
        frame.groupby("task_name", dropna=False)
        .agg(
            runs=("task_name", "size"),
            configs=("config_name", "nunique"),
            success_rate=("success", "mean"),
            solved_anywhere=("success", "any"),
            solved_everywhere=("success", "all"),
            exception_rate=("exception_seen", "mean"),
        )
        .reset_index()
        .sort_values(["success_rate", "runs"], ascending=[True, False])
    )
    return _round_numeric(summary)


def build_wolfbench_task_config_summary(task_run_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-task consistency within each WolfBench configuration."""
    if task_run_df.empty:
        return pd.DataFrame(
            columns=[
                "config_name",
                "agent",
                "model",
                "task_name",
                "runs",
                "success_rate",
                "solved_any_run",
                "solved_all_runs",
                "exception_rate",
            ]
        )
    frame = task_run_df.copy()
    frame["exception_seen"] = frame["exception_type"].notna()
    summary = (
        frame.groupby(["config_name", "agent", "model", "task_name"], dropna=False)
        .agg(
            runs=("task_name", "size"),
            success_rate=("success", "mean"),
            solved_any_run=("success", "any"),
            solved_all_runs=("success", "all"),
            exception_rate=("exception_seen", "mean"),
        )
        .reset_index()
        .sort_values(
            ["config_name", "success_rate", "task_name"],
            ascending=[True, True, True],
        )
    )
    return _round_numeric(summary)


def build_wolfbench_consistency_contrast_summary(
    config_summary: pd.DataFrame,
    *,
    min_runs: int,
) -> pd.DataFrame:
    """Compare configurations that share the same displayed model name."""
    if config_summary.empty:
        return pd.DataFrame()
    eligible = config_summary[config_summary["runs"] >= min_runs].copy()
    rows: list[dict[str, Any]] = []
    for model_display, group in eligible.groupby("model_display", dropna=False):
        records = group.sort_values("config_name").to_dict("records")
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                rows.append(
                    {
                        "model_display": model_display,
                        "config_a": left["config_name"],
                        "config_b": right["config_name"],
                        "agent_a": left.get("agent"),
                        "agent_b": right.get("agent"),
                        "runs_a": left.get("runs"),
                        "runs_b": right.get("runs"),
                        "average_a": left.get("average_score"),
                        "average_b": right.get("average_score"),
                        "average_gap": _abs_gap(
                            left.get("average_score"), right.get("average_score")
                        ),
                        "solid_a": left.get("solid_score"),
                        "solid_b": right.get("solid_score"),
                        "solid_gap": _abs_gap(left.get("solid_score"), right.get("solid_score")),
                        "ceiling_a": left.get("ceiling_score"),
                        "ceiling_b": right.get("ceiling_score"),
                        "ceiling_gap": _abs_gap(
                            left.get("ceiling_score"), right.get("ceiling_score")
                        ),
                        "spread_a": left.get("score_spread"),
                        "spread_b": right.get("score_spread"),
                        "spread_gap": _abs_gap(
                            left.get("score_spread"), right.get("score_spread")
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_display",
                "config_a",
                "config_b",
                "average_gap",
                "solid_gap",
                "ceiling_gap",
                "spread_gap",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["contrast_score"] = (
        frame["average_gap"].fillna(0)
        + frame["solid_gap"].fillna(0)
        + frame["ceiling_gap"].fillna(0)
        + frame["spread_gap"].fillna(0)
    )
    return _round_numeric(frame.sort_values("contrast_score", ascending=False))


def _iter_run_refs(source: str | Path, *, max_runs: int | None) -> list[_RunRef]:
    source_text = str(source)
    if source_text.startswith("http://") or source_text.startswith("https://"):
        return _iter_github_run_refs(max_runs=max_runs)
    return _iter_local_run_refs(Path(source), max_runs=max_runs)


def _iter_local_run_refs(source: Path, *, max_runs: int | None) -> list[_RunRef]:
    runs_root = source / "wolfbench-runs" if (source / "wolfbench-runs").exists() else source
    if not runs_root.exists():
        raise FileNotFoundError(f"WolfBench source does not exist: {runs_root}")

    refs: list[_RunRef] = []
    for config_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in config_dir.iterdir() if path.is_dir()):
            result_path = run_dir / "result.json"
            if not result_path.exists():
                continue
            refs.append(
                _RunRef(
                    config_name=config_dir.name,
                    run_name=run_dir.name,
                    result_uri=result_path,
                    config_uri=run_dir / "config.json",
                )
            )
            if max_runs is not None and len(refs) >= max_runs:
                return refs
    return refs


def _iter_github_run_refs(*, max_runs: int | None) -> list[_RunRef]:
    refs: list[_RunRef] = []
    for config_item in _load_github_dir(_GITHUB_API_RUNS_URL):
        if config_item.get("type") != "dir":
            continue
        config_name = str(config_item["name"])
        encoded_config = quote(config_name, safe="")
        api_url = f"{_GITHUB_API_RUNS_PATH}/{encoded_config}?ref=main"
        raw_base = f"{_RAW_RUNS_BASE_URL}/{quote(config_name, safe='')}"
        for run_item in _load_github_dir(api_url):
            if run_item.get("type") != "dir":
                continue
            run_name = str(run_item["name"])
            encoded_run = quote(run_name, safe="")
            refs.append(
                _RunRef(
                    config_name=config_name,
                    run_name=run_name,
                    result_uri=f"{raw_base}/{encoded_run}/result.json",
                    config_uri=f"{raw_base}/{encoded_run}/config.json",
                )
            )
            if max_runs is not None and len(refs) >= max_runs:
                return refs
    return refs


def _records_from_run(
    run_ref: _RunRef,
    result: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = result.get("stats") or {}
    evals = stats.get("evals") or {}
    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for eval_key, eval_payload in evals.items():
        if not isinstance(eval_payload, dict):
            continue
        parsed_key = _parse_eval_key(str(eval_key))
        agent = _agent_name(config) or parsed_key["agent"]
        model = _model_name(config) or parsed_key["model"]
        dataset = parsed_key["dataset"]
        exception_by_trial = _exception_by_trial(eval_payload.get("exception_stats"))
        reward_rows = _reward_rows(eval_payload.get("reward_stats"))
        solved_tasks = sum(1 for row in reward_rows if row["success"])
        failed_tasks = sum(1 for row in reward_rows if not row["success"])
        exception_types = sorted(
            {exception for exception in exception_by_trial.values() if exception}
        )
        run_key = f"{run_ref.run_key}/{eval_key}"
        started_at = result.get("started_at")
        finished_at = result.get("finished_at")

        run_rows.append(
            {
                "run_key": run_key,
                "config_name": run_ref.config_name,
                "run_name": run_ref.run_name,
                "eval_key": eval_key,
                "job_id": result.get("id"),
                "agent": agent,
                "model": model,
                "model_display": config.get("model_display") or _display_model(model),
                "thinking_display": config.get("thinking_display") or "",
                "dataset": dataset,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": _duration_seconds(started_at, finished_at),
                "n_total_trials": result.get("n_total_trials"),
                "n_trials": eval_payload.get("n_trials", stats.get("n_trials")),
                "n_errors": eval_payload.get("n_errors", stats.get("n_errors")),
                "score_mean": _metric_mean(eval_payload),
                "solved_tasks": solved_tasks,
                "failed_tasks": failed_tasks,
                "reward_task_count": len(reward_rows),
                "exception_count": len(exception_by_trial),
                "exception_types": ",".join(exception_types),
            }
        )

        for row in reward_rows:
            task_rows.append(
                {
                    "run_key": run_key,
                    "config_name": run_ref.config_name,
                    "run_name": run_ref.run_name,
                    "eval_key": eval_key,
                    "agent": agent,
                    "model": model,
                    "model_display": config.get("model_display") or _display_model(model),
                    "thinking_display": config.get("thinking_display") or "",
                    "dataset": dataset,
                    "task_name": _trial_to_task_name(row["trial_name"]),
                    "trial_name": row["trial_name"],
                    "reward": row["reward"],
                    "success": row["success"],
                    "exception_type": exception_by_trial.get(row["trial_name"]),
                }
            )

    return run_rows, task_rows


def _reward_rows(reward_stats: object) -> list[dict[str, Any]]:
    if not isinstance(reward_stats, dict):
        return []
    reward_buckets = reward_stats.get("reward")
    if not isinstance(reward_buckets, dict):
        return []
    rows: list[dict[str, Any]] = []
    for reward_raw, trial_names in reward_buckets.items():
        try:
            reward = float(reward_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(trial_names, list):
            continue
        for trial_name in trial_names:
            rows.append(
                {
                    "trial_name": str(trial_name),
                    "reward": reward,
                    "success": reward > 0.0,
                }
            )
    return rows


def _exception_by_trial(exception_stats: object) -> dict[str, str]:
    if not isinstance(exception_stats, dict):
        return {}
    result: dict[str, str] = {}
    for exception_type, trial_names in exception_stats.items():
        if not isinstance(trial_names, list):
            continue
        for trial_name in trial_names:
            result[str(trial_name)] = str(exception_type)
    return result


def _success_pivot(task_group: pd.DataFrame) -> pd.DataFrame:
    if task_group.empty:
        return pd.DataFrame()
    frame = task_group[["task_name", "run_key", "success"]].copy()
    frame["success"] = frame["success"].astype(bool)
    return frame.pivot_table(
        index="task_name",
        columns="run_key",
        values="success",
        aggfunc="max",
        fill_value=False,
    ).astype(bool)


def _parse_eval_key(eval_key: str) -> dict[str, str]:
    parts = eval_key.split("__")
    return {
        "agent": parts[0] if len(parts) > 0 else "unknown",
        "model": parts[1] if len(parts) > 1 else "unknown",
        "dataset": parts[2] if len(parts) > 2 else "unknown",
    }


def _agent_name(config: dict[str, Any]) -> str | None:
    agents = config.get("agents")
    if isinstance(agents, list) and agents and isinstance(agents[0], dict):
        name = agents[0].get("name")
        return str(name) if name else None
    return None


def _model_name(config: dict[str, Any]) -> str | None:
    agents = config.get("agents")
    if isinstance(agents, list) and agents and isinstance(agents[0], dict):
        name = agents[0].get("model_name")
        return str(name) if name else None
    return None


def _display_model(model: object) -> str:
    value = str(model or "unknown")
    return value.rsplit("/", maxsplit=1)[-1]


def _metric_mean(eval_payload: dict[str, Any]) -> float | None:
    metrics = eval_payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return None
    metric = metrics[0]
    if not isinstance(metric, dict):
        return None
    try:
        return float(metric.get("mean"))
    except (TypeError, ValueError):
        return None


def _trial_to_task_name(trial_name: str) -> str:
    return trial_name.split("__", maxsplit=1)[0]


def _duration_seconds(started_at: object, finished_at: object) -> float | None:
    start = _parse_datetime(started_at)
    finish = _parse_datetime(finished_at)
    if start is None or finish is None:
        return None
    return (finish - start).total_seconds()


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_github_dir(url: str) -> list[dict[str, Any]]:
    payload = _load_json(url)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "message" in payload:
        message = payload.get("message")
        raise RuntimeError(f"GitHub API error while reading WolfBench: {message}")
    raise RuntimeError(f"Unexpected GitHub directory response for {url}")


def _load_json(uri: str | Path, *, required: bool = True) -> Any:
    if isinstance(uri, Path):
        if not uri.exists():
            if required:
                raise FileNotFoundError(uri)
            return None
        return json.loads(uri.read_text())

    request = Request(uri, headers=_request_headers())
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if not required and exc.code == 404:
            return None
        if exc.code in {403, 429}:
            raise RuntimeError(
                "GitHub refused the WolfBench request. Set GITHUB_TOKEN or use a "
                "local WolfBench checkout for full-corpus reports."
            ) from exc
        raise
    except URLError as exc:
        raise RuntimeError(f"Unable to read WolfBench source {uri}: {exc}") from exc


def _request_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "blast-radius-bench",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _write_plots(
    *,
    config_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    output_dir: Path,
    min_runs: int,
) -> list[Path]:
    plot_paths: list[Path] = []
    eligible = config_summary[config_summary["runs"] >= min_runs].copy()
    if not eligible.empty:
        plot_paths.append(
            _plot_five_metric_bars(
                eligible.sort_values("average_score", ascending=False).head(16),
                output_dir / "config_five_metrics.png",
            )
        )
        plot_paths.append(
            _plot_ceiling_solid_scatter(
                eligible,
                output_dir / "config_ceiling_vs_solid.png",
            )
        )
        plot_paths.append(
            _plot_horizontal_bar(
                eligible.sort_values("ceiling_minus_solid", ascending=False).head(20),
                value_col="ceiling_minus_solid",
                label_col="config_name",
                title="Largest Ceiling-minus-Solid Consistency Gaps",
                xlabel="Ceiling - Solid",
                output_path=output_dir / "config_consistency_gap.png",
            )
        )

    if not task_summary.empty:
        plot_paths.append(
            _plot_horizontal_bar(
                task_summary.sort_values(["success_rate", "runs"], ascending=[True, False]).head(30),
                value_col="success_rate",
                label_col="task_name",
                title="Hardest WolfBench Tasks by Success Rate",
                xlabel="Success Rate",
                output_path=output_dir / "task_success_rate.png",
                percent_axis=True,
            )
        )

    return plot_paths


def _plot_five_metric_bars(frame: pd.DataFrame, output_path: Path) -> Path:
    metric_cols = [
        "solid_score",
        "worst_of_score",
        "average_score",
        "best_of_score",
        "ceiling_score",
    ]
    labels = ["Solid", "Worst", "Average", "Best", "Ceiling"]
    plot_frame = frame.set_index("config_name")[metric_cols].iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_frame) * 0.48)))
    plot_frame.plot(kind="barh", ax=ax, width=0.82)
    ax.set_title("WolfBench Five-Metric Profile by Configuration")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.legend(labels=labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_ceiling_solid_scatter(frame: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = frame["runs"].clip(lower=1) * 24
    ax.scatter(
        frame["solid_score"],
        frame["ceiling_score"],
        s=sizes,
        alpha=0.68,
        color="#2e6f9e",
    )
    for _, row in frame.sort_values("average_score", ascending=False).head(12).iterrows():
        ax.annotate(row["config_name"], (row["solid_score"], row["ceiling_score"]), fontsize=8)
    ax.plot([0, 1], [0, 1], color="#777", linewidth=1, linestyle="--")
    ax.set_title("Ceiling vs Solid Reliability")
    ax.set_xlabel("Solid score")
    ax.set_ylabel("Ceiling score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_horizontal_bar(
    frame: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    percent_axis: bool = False,
) -> Path:
    sorted_frame = frame.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_frame) * 0.38)))
    ax.barh(sorted_frame[label_col], sorted_frame[value_col], color="#1d7f64")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if percent_axis:
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_html_report(
    *,
    source: str,
    output_dir: Path,
    run_df: pd.DataFrame,
    task_run_df: pd.DataFrame,
    config_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    consistency_contrast_summary: pd.DataFrame,
    plot_paths: list[Path],
    min_runs: int,
) -> Path:
    output_path = output_dir / "index.html"
    best_config = (
        str(config_summary.iloc[0]["config_name"]) if not config_summary.empty else "n/a"
    )
    headline_stats = {
        "Runs": f"{len(run_df):,}",
        "Task rows": f"{len(task_run_df):,}",
        "Configs": f"{run_df['config_name'].nunique():,}" if not run_df.empty else "0",
        "Tasks": f"{task_run_df['task_name'].nunique():,}" if not task_run_df.empty else "0",
        "Mean score": f"{run_df['score_mean'].mean():.1%}" if not run_df.empty else "n/a",
        "Best config": best_config,
    }
    stats_html = "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in headline_stats.items()
    )
    plot_lookup = {path.stem: path.name for path in plot_paths}
    plot_html = _plot_grid_html(
        plot_lookup,
        [
            ("config_five_metrics", "Five-Metric Profiles"),
            ("config_ceiling_vs_solid", "Ceiling vs Solid"),
            ("config_consistency_gap", "Consistency Gaps"),
            ("task_success_rate", "Hardest Tasks"),
        ],
    )
    config_table = _dashboard_table(
        config_summary,
        [
            "config_name",
            "agent",
            "model_display",
            "thinking_display",
            "runs",
            "average_score",
            "solid_score",
            "ceiling_score",
            "ceiling_minus_solid",
            "mean_errors",
        ],
        limit=40,
        table_id="config-summary-table",
    )
    contrast_table = _dashboard_table(
        consistency_contrast_summary,
        [
            "model_display",
            "config_a",
            "config_b",
            "average_gap",
            "solid_gap",
            "ceiling_gap",
            "spread_gap",
            "contrast_score",
        ],
        limit=30,
    )
    task_table = _dashboard_table(
        task_summary,
        ["task_name", "runs", "configs", "success_rate", "exception_rate"],
        limit=40,
        table_id="task-summary-table",
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WolfBench Report</title>
  <style>
    :root {{
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #62625f;
      --line: #dedbd2;
      --green: #1d7f64;
      --blue: #2e6f9e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); line-height: 1.45; }}
    .shell {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
    header {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 0.45fr); gap: 22px; align-items: end; padding: 28px 0 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: clamp(34px, 5vw, 68px); line-height: 0.95; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    code {{ background: #ece9df; padding: 2px 6px; border-radius: 4px; }}
    .lede {{ max-width: 860px; margin-top: 14px; font-size: 18px; color: #343431; }}
    .meta {{ display: grid; gap: 8px; align-content: end; font-size: 14px; min-width: 0; }}
    .meta div {{ overflow-wrap: anywhere; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 20px 0 28px; }}
    .metric {{ min-height: 82px; border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px 14px; display: grid; align-content: space-between; }}
    .metric span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric strong {{ font-size: 24px; line-height: 1; overflow-wrap: anywhere; }}
    .band {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .plot-card {{ border: 1px solid var(--line); background: #fff; border-radius: 8px; padding: 12px; }}
    .plot-card img {{ width: 100%; display: block; border-radius: 6px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #ece9df; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f4f1e8; z-index: 1; color: #3a3935; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
    tr:hover td {{ background: #fbfaf6; }}
    .filter {{ width: min(420px, 100%); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; margin-bottom: 10px; }}
    @media (max-width: 980px) {{ .shell {{ padding: 16px; }} header, .grid-2, .plot-grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>WolfBench Readout</h1>
        <p class="lede">Published Terminal-Bench 2.0 runs summarized through WolfBench's five-metric reliability lens. This uses GitHub run-level JSON; trace-level blast-radius scoring needs the Weave artifacts.</p>
      </div>
      <div class="meta">
        <div>Source <code>{html.escape(source)}</code></div>
        <div>Min runs for plots <code>{min_runs}</code></div>
      </div>
    </header>
    <section class="metrics">{stats_html}</section>
    <section class="band">{plot_html}</section>
    <section class="grid-2">
      <div class="band"><h2>Configuration Summary</h2><input class="filter" data-target="config-summary-table" placeholder="Filter configs" />{config_table}</div>
      <div class="band"><h2>Same-Model Contrasts</h2>{contrast_table}</div>
    </section>
    <section class="band"><h2>Task Difficulty</h2><input class="filter" data-target="task-summary-table" placeholder="Filter tasks" />{task_table}</section>
  </div>
  <script>
    document.querySelectorAll('.filter').forEach((input) => {{
      const table = document.getElementById(input.dataset.target);
      if (!table) return;
      input.addEventListener('input', () => {{
        const query = input.value.toLowerCase();
        table.querySelectorAll('tbody tr').forEach((row) => {{
          row.style.display = row.innerText.toLowerCase().includes(query) ? '' : 'none';
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_doc)
    return output_path


def _plot_grid_html(plot_lookup: dict[str, str], plots: list[tuple[str, str]]) -> str:
    cards = []
    for stem, title in plots:
        filename = plot_lookup.get(stem)
        if not filename:
            continue
        cards.append(
            "<article class='plot-card'>"
            f"<h3>{html.escape(title)}</h3>"
            f"<img src='{html.escape(filename)}' alt='{html.escape(title)}' />"
            "</article>"
        )
    if not cards:
        return "<p>No plots available.</p>"
    return f"<div class='plot-grid'>{''.join(cards)}</div>"


def _dashboard_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    limit: int,
    table_id: str | None = None,
) -> str:
    if frame.empty:
        return "<p>No rows available.</p>"
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return "<p>No selected columns available.</p>"
    rendered = frame[selected].head(limit).copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(_format_cell)
    return (
        "<div class='table-wrap'>"
        + rendered.to_html(index=False, escape=False, border=0, table_id=table_id)
        .replace('class="dataframe"', 'class="dataframe dashboard-table"')
        + "</div>"
    )


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if -1 <= value <= 1:
            return f"{value:.3f}"
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_cols = result.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_cols:
        if result[column].dtype == bool:
            continue
        result[column] = result[column].round(4)
    return result


def _abs_gap(left: object, right: object) -> float | None:
    try:
        if pd.isna(left) or pd.isna(right):
            return None
        return abs(float(left) - float(right))
    except (TypeError, ValueError):
        return None
