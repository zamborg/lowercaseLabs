"""Utilities for analyzing public Terminal-Bench trajectory datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PUBLIC_TERMINALBENCH_DATASETS: dict[str, str] = {
    "yoonholee/terminalbench-trajectories": (
        "https://huggingface.co/datasets/yoonholee/terminalbench-trajectories"
    ),
    "hanspeterlyngsoeraaschoujensen/terminal-bench-sample-eval-trajectories": (
        "https://huggingface.co/datasets/"
        "hanspeterlyngsoeraaschoujensen/terminal-bench-sample-eval-trajectories"
    ),
    "hanspeterlyngsoeraaschoujensen/terminal-bench-pro-eval-trajectories": (
        "https://huggingface.co/datasets/"
        "hanspeterlyngsoeraaschoujensen/terminal-bench-pro-eval-trajectories"
    ),
}

_SHELL_TOOL_HINTS = ("bash", "shell", "run_shell", "execute_bash")
_READ_TOOL_HINTS = ("read", "open", "cat", "grep")
_EDIT_TOOL_HINTS = ("edit", "write", "replace", "patch")
_PLAN_TOOL_HINTS = ("todo", "task_tracker", "think", "plan")


def build_public_terminalbench_report(
    dataset_id: str,
    output_dir: str | Path,
    *,
    split: str = "train",
    max_rows: int | None = None,
    min_runs: int = 100,
    top_agents: int = 12,
) -> dict[str, Any]:
    """Build a static report for a public Terminal-Bench trajectory dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_id, split=split)
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    run_df, agent_tool_df, agent_tool_category_df = _summarize_dataset(dataset, dataset_id)

    agent_summary = _build_agent_summary(run_df)
    model_summary = _build_model_summary(run_df, min_runs=max(10, min_runs // 4))
    task_summary = _build_task_summary(run_df)
    task_agent_summary = _build_task_agent_summary(run_df)
    tool_summary = _build_tool_summary(agent_tool_df)
    focus_task = _select_focus_task(task_summary)

    run_df.to_csv(output_path / "run_metrics.csv", index=False)
    agent_summary.to_csv(output_path / "agent_summary.csv", index=False)
    model_summary.to_csv(output_path / "model_summary.csv", index=False)
    task_summary.to_csv(output_path / "task_summary.csv", index=False)
    task_agent_summary.to_csv(output_path / "task_agent_summary.csv", index=False)
    tool_summary.to_csv(output_path / "tool_summary.csv", index=False)
    agent_tool_category_df.to_csv(output_path / "tool_category_summary.csv", index=False)

    plot_paths = _write_plots(
        run_df=run_df,
        agent_summary=agent_summary,
        task_summary=task_summary,
        task_agent_summary=task_agent_summary,
        agent_tool_category_df=agent_tool_category_df,
        output_dir=output_path,
        min_runs=min_runs,
        top_agents=top_agents,
        focus_task=focus_task,
    )

    html_path = _write_html_report(
        dataset_id=dataset_id,
        output_dir=output_path,
        run_df=run_df,
        agent_summary=agent_summary,
        model_summary=model_summary,
        task_summary=task_summary,
        tool_summary=tool_summary,
        plot_paths=plot_paths,
        min_runs=min_runs,
        focus_task=focus_task,
    )

    return {
        "dataset_id": dataset_id,
        "dataset_url": PUBLIC_TERMINALBENCH_DATASETS.get(dataset_id),
        "rows": int(len(run_df)),
        "agents": int(run_df["agent"].nunique()),
        "models": int(run_df["model"].nunique()),
        "tasks": int(run_df["task_name"].nunique()),
        "focus_task": focus_task,
        "output_dir": str(output_path),
        "html_report": str(html_path),
        "plots": [str(path) for path in plot_paths],
        "known_public_datasets": [
            {"dataset_id": key, "url": value}
            for key, value in sorted(PUBLIC_TERMINALBENCH_DATASETS.items())
        ],
    }


def _summarize_dataset(dataset: Any, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    agent_tool_counter: dict[str, Counter[str]] = defaultdict(Counter)
    agent_tool_category_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for row in dataset:
        steps_raw = row.get("steps")
        has_steps = _has_steps(steps_raw)
        step_summary = summarize_public_steps(steps_raw) if has_steps else _empty_step_summary()
        agent = row.get("agent") or "unknown"

        records.append(
            {
                "dataset_id": dataset_id,
                "task_name": row.get("task_name") or row.get("instance_id"),
                "agent": agent,
                "model": row.get("model") or "unknown",
                "reward": _coerce_float(row.get("reward")),
                "success": _coerce_success(row.get("reward")),
                "duration_seconds": _coerce_float(row.get("duration_seconds")),
                "input_tokens": _coerce_float(row.get("input_tokens")),
                "output_tokens": _coerce_float(row.get("output_tokens")),
                "cache_tokens": _coerce_float(row.get("cache_tokens")),
                "cost_cents": _coerce_float(row.get("cost_cents")),
                "trial_name": row.get("trial_name"),
                "trial_id": row.get("trial_id"),
                "has_steps": has_steps,
                "step_count": step_summary["step_count"],
                "agent_step_count": step_summary["agent_step_count"],
                "tool_call_count": step_summary["tool_call_count"],
                "shell_tool_calls": step_summary["shell_tool_calls"],
                "read_tool_calls": step_summary["read_tool_calls"],
                "edit_tool_calls": step_summary["edit_tool_calls"],
                "plan_tool_calls": step_summary["plan_tool_calls"],
            }
        )

        if has_steps:
            agent_tool_counter[agent].update(step_summary["tool_counter"])
            agent_tool_category_counter[agent].update(step_summary["tool_category_counter"])

    run_df = pd.DataFrame.from_records(records)
    run_df["success"] = run_df["success"].fillna(False).astype(bool)

    tool_rows = []
    for agent, counter in agent_tool_counter.items():
        for tool_name, count in counter.items():
            tool_rows.append({"agent": agent, "tool_name": tool_name, "count": count})
    agent_tool_df = pd.DataFrame(tool_rows)
    if agent_tool_df.empty:
        agent_tool_df = pd.DataFrame(columns=["agent", "tool_name", "count"])

    category_rows = []
    for agent, counter in agent_tool_category_counter.items():
        total = sum(counter.values())
        for category, count in counter.items():
            category_rows.append(
                {
                    "agent": agent,
                    "tool_category": category,
                    "count": count,
                    "share": (count / total) if total else 0.0,
                }
            )
    agent_tool_category_df = pd.DataFrame(category_rows)
    if agent_tool_category_df.empty:
        agent_tool_category_df = pd.DataFrame(
            columns=["agent", "tool_category", "count", "share"]
        )

    return run_df, agent_tool_df, agent_tool_category_df


def summarize_public_steps(steps_raw: str | None) -> dict[str, Any]:
    """Summarize one public Terminal-Bench trace blob."""
    if not _has_steps(steps_raw):
        return _empty_step_summary()

    try:
        steps = json.loads(steps_raw or "[]")
    except json.JSONDecodeError:
        return _empty_step_summary()

    tool_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    tool_call_count = 0
    agent_step_count = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("src") == "agent":
            agent_step_count += 1
        tools = step.get("tools") or []
        for tool in tools:
            tool_name = _public_tool_name(tool)
            if tool_name is None:
                continue
            tool_call_count += 1
            tool_counter[tool_name] += 1
            category_counter[classify_public_tool(tool_name)] += 1

    return {
        "step_count": len(steps),
        "agent_step_count": agent_step_count,
        "tool_call_count": tool_call_count,
        "shell_tool_calls": sum(
            count for name, count in tool_counter.items() if classify_public_tool(name) == "shell"
        ),
        "read_tool_calls": sum(
            count for name, count in tool_counter.items() if classify_public_tool(name) == "read"
        ),
        "edit_tool_calls": sum(
            count for name, count in tool_counter.items() if classify_public_tool(name) == "edit"
        ),
        "plan_tool_calls": sum(
            count for name, count in tool_counter.items() if classify_public_tool(name) == "plan"
        ),
        "tool_counter": tool_counter,
        "tool_category_counter": category_counter,
    }


def classify_public_tool(tool_name: str) -> str:
    """Map a public trajectory tool name to a coarse behavioral category."""
    lower_name = tool_name.lower()
    if any(hint in lower_name for hint in _SHELL_TOOL_HINTS):
        return "shell"
    if any(hint in lower_name for hint in _PLAN_TOOL_HINTS):
        return "plan"
    if any(hint in lower_name for hint in _READ_TOOL_HINTS):
        return "read"
    if any(hint in lower_name for hint in _EDIT_TOOL_HINTS):
        return "edit"
    return "other"


def _public_tool_name(tool: object) -> str | None:
    if isinstance(tool, str):
        return tool
    if not isinstance(tool, dict):
        return None
    for key in ("fn", "function_name", "name", "tool_name", "function"):
        value = tool.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_agent_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        run_df.groupby("agent", dropna=False)
        .agg(
            runs=("agent", "size"),
            success_rate=("success", "mean"),
            mean_reward=("reward", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_cost_cents=("cost_cents", "mean"),
            trace_coverage=("has_steps", "mean"),
            mean_step_count=("step_count", "mean"),
            mean_tool_call_count=("tool_call_count", "mean"),
            mean_shell_tool_calls=("shell_tool_calls", "mean"),
            mean_read_tool_calls=("read_tool_calls", "mean"),
            mean_edit_tool_calls=("edit_tool_calls", "mean"),
        )
        .reset_index()
        .sort_values(["runs", "success_rate"], ascending=[False, False])
    )
    return _round_numeric(summary)


def _build_model_summary(run_df: pd.DataFrame, *, min_runs: int) -> pd.DataFrame:
    summary = (
        run_df.groupby(["agent", "model"], dropna=False)
        .agg(
            runs=("model", "size"),
            success_rate=("success", "mean"),
            mean_reward=("reward", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            mean_cost_cents=("cost_cents", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            trace_coverage=("has_steps", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["runs"] >= min_runs].sort_values(
        ["success_rate", "runs"],
        ascending=[False, False],
    )
    return _round_numeric(summary)


def _build_tool_summary(agent_tool_df: pd.DataFrame) -> pd.DataFrame:
    if agent_tool_df.empty:
        return pd.DataFrame(columns=["tool_name", "count"])
    summary = (
        agent_tool_df.groupby("tool_name", dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return summary


def _build_task_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        run_df.groupby("task_name", dropna=False)
        .agg(
            runs=("task_name", "size"),
            success_rate=("success", "mean"),
            mean_reward=("reward", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            trace_coverage=("has_steps", "mean"),
            traced_runs=("has_steps", "sum"),
            mean_tool_call_count=("tool_call_count", "mean"),
        )
        .reset_index()
        .sort_values(["runs", "success_rate"], ascending=[False, False])
    )
    return _round_numeric(summary)


def _build_task_agent_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        run_df.groupby(["task_name", "agent"], dropna=False)
        .agg(
            runs=("agent", "size"),
            success_rate=("success", "mean"),
            trace_coverage=("has_steps", "mean"),
            mean_tool_call_count=("tool_call_count", "mean"),
        )
        .reset_index()
    )
    return _round_numeric(summary)


def _select_focus_task(task_summary: pd.DataFrame) -> str | None:
    if task_summary.empty:
        return None
    ordered = task_summary.sort_values(
        ["traced_runs", "runs", "success_rate"],
        ascending=[False, False, True],
    )
    return str(ordered.iloc[0]["task_name"])


def _write_plots(
    *,
    run_df: pd.DataFrame,
    agent_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    task_agent_summary: pd.DataFrame,
    agent_tool_category_df: pd.DataFrame,
    output_dir: Path,
    min_runs: int,
    top_agents: int,
    focus_task: str | None,
) -> list[Path]:
    plot_paths: list[Path] = []

    top_agent_summary = agent_summary.head(top_agents).copy()
    plot_paths.append(
        _plot_horizontal_bar(
            top_agent_summary,
            value_col="runs",
            label_col="agent",
            title=f"Top {len(top_agent_summary)} Agents by Rollout Count",
            xlabel="Runs",
            output_path=output_dir / "agent_runs.png",
        )
    )

    eligible = agent_summary[agent_summary["runs"] >= min_runs].copy()
    if not eligible.empty:
        plot_paths.append(
            _plot_horizontal_bar(
                eligible.sort_values("success_rate", ascending=False).head(top_agents),
                value_col="success_rate",
                label_col="agent",
                title=f"Success Rate by Agent (min {min_runs} runs)",
                xlabel="Success Rate",
                output_path=output_dir / "agent_success_rate.png",
                percent_axis=True,
            )
        )
        plot_paths.append(
            _plot_scatter(
                eligible,
                x_col="mean_cost_cents",
                y_col="success_rate",
                size_col="runs",
                label_col="agent",
                title=f"Success Rate vs Mean Cost (min {min_runs} runs)",
                xlabel="Mean Cost (cents)",
                ylabel="Success Rate",
                output_path=output_dir / "agent_success_vs_cost.png",
            )
        )
        plot_paths.append(
            _plot_horizontal_bar(
                eligible.sort_values("median_duration_seconds", ascending=False).head(top_agents),
                value_col="median_duration_seconds",
                label_col="agent",
                title=f"Median Duration by Agent (min {min_runs} runs)",
                xlabel="Median Duration (s)",
                output_path=output_dir / "agent_duration.png",
            )
        )
        plot_paths.append(
            _plot_horizontal_bar(
                eligible.sort_values("mean_tool_call_count", ascending=False).head(top_agents),
                value_col="mean_tool_call_count",
                label_col="agent",
                title=f"Mean Tool Calls per Run (min {min_runs} runs)",
                xlabel="Mean Tool Calls",
                output_path=output_dir / "agent_tool_calls.png",
            )
        )
        plot_paths.append(
            _plot_horizontal_bar(
                eligible.sort_values("trace_coverage", ascending=False).head(top_agents),
                value_col="trace_coverage",
                label_col="agent",
                title=f"Trace Coverage by Agent (min {min_runs} runs)",
                xlabel="Fraction of Runs with Embedded Steps",
                output_path=output_dir / "agent_trace_coverage.png",
                percent_axis=True,
            )
        )

    if not task_summary.empty:
        plot_paths.append(
            _plot_horizontal_bar(
                task_summary.sort_values(["success_rate", "runs"], ascending=[True, False]).head(25),
                value_col="success_rate",
                label_col="task_name",
                title="Hardest Tasks by Success Rate",
                xlabel="Success Rate",
                output_path=output_dir / "task_success_rate.png",
                percent_axis=True,
            )
        )

    if not task_agent_summary.empty and not agent_summary.empty:
        top_agent_names = (
            agent_summary.head(min(top_agents, len(agent_summary)))["agent"].tolist()
        )
        heatmap_df = task_agent_summary[task_agent_summary["agent"].isin(top_agent_names)].copy()
        if not heatmap_df.empty:
            plot_paths.append(
                _plot_task_agent_heatmap(
                    heatmap_df,
                    output_path=output_dir / "task_agent_success_heatmap.png",
                )
            )

    if focus_task is not None:
        focus_df = run_df[(run_df["task_name"] == focus_task) & (run_df["has_steps"])].copy()
        if not focus_df.empty:
            plot_paths.append(
                _plot_focus_task_histogram(
                    focus_df,
                    task_name=focus_task,
                    output_path=output_dir / f"focus_task_tool_histogram_{_slugify(focus_task)}.png",
                )
            )

    if not agent_tool_category_df.empty:
        traced_counts = (
            run_df.groupby("agent", dropna=False)["has_steps"].sum().reset_index(name="traced_runs")
        )
        top_traced_agents = traced_counts.sort_values("traced_runs", ascending=False).head(8)["agent"]
        mix_df = agent_tool_category_df[
            agent_tool_category_df["agent"].isin(top_traced_agents)
        ].copy()
        if not mix_df.empty:
            plot_paths.append(
                _plot_tool_mix(
                    mix_df,
                    output_path=output_dir / "agent_tool_mix.png",
                )
            )

    return plot_paths


def _write_html_report(
    *,
    dataset_id: str,
    output_dir: Path,
    run_df: pd.DataFrame,
    agent_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    tool_summary: pd.DataFrame,
    plot_paths: list[Path],
    min_runs: int,
    focus_task: str | None,
) -> Path:
    dataset_url = PUBLIC_TERMINALBENCH_DATASETS.get(dataset_id, "")
    output_path = output_dir / "index.html"

    headline_stats = {
        "Runs": f"{len(run_df):,}",
        "Agents": f"{run_df['agent'].nunique():,}",
        "Models": f"{run_df['model'].nunique():,}",
        "Tasks": f"{run_df['task_name'].nunique():,}",
        "Success rate": f"{run_df['success'].mean():.1%}",
        "Trace coverage": f"{run_df['has_steps'].mean():.1%}",
    }

    stats_html = "".join(
        f"<div class='stat'><span class='label'>{label}</span><span class='value'>{value}</span></div>"
        for label, value in headline_stats.items()
    )
    plot_html = "".join(
        f"<section class='plot'><h2>{path.stem.replace('_', ' ').title()}</h2>"
        f"<img src='{path.name}' alt='{path.stem}' /></section>"
        for path in plot_paths
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Terminal-Bench Public Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem auto; max-width: 1100px; line-height: 1.5; color: #1f2328; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    p {{ margin-top: 0.2rem; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin: 1.5rem 0; }}
    .stat {{ border: 1px solid #d0d7de; border-radius: 10px; padding: 0.8rem 1rem; background: #f6f8fa; }}
    .label {{ display: block; font-size: 0.85rem; color: #57606a; }}
    .value {{ display: block; font-size: 1.4rem; font-weight: 700; }}
    .plot img {{ width: 100%; border: 1px solid #d0d7de; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.55rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .muted {{ color: #57606a; }}
    code {{ background: #f6f8fa; padding: 0.05rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Terminal-Bench Public Rollout Report</h1>
  <p>Dataset: <code>{dataset_id}</code></p>
  <p class="muted">Source: <a href="{dataset_url}">{dataset_url}</a></p>
  <p class="muted">Agents with fewer than {min_runs} runs are excluded from some aggregate plots.</p>
  <p class="muted">Focus task for the per-task histogram: <code>{focus_task or "n/a"}</code></p>
  <div class="stats">{stats_html}</div>
  {plot_html}
  <section>
    <h2>Agent Summary</h2>
    {agent_summary.head(20).to_html(index=False)}
  </section>
  <section>
    <h2>Model Summary</h2>
    {model_summary.head(30).to_html(index=False)}
  </section>
  <section>
    <h2>Task Summary</h2>
    {task_summary.sort_values(['success_rate', 'runs'], ascending=[True, False]).head(30).to_html(index=False)}
  </section>
  <section>
    <h2>Top Tools</h2>
    {tool_summary.head(30).to_html(index=False)}
  </section>
</body>
</html>
"""
    output_path.write_text(html)
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
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_frame) * 0.45)))
    ax.barh(sorted_frame[label_col], sorted_frame[value_col], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if percent_axis:
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_scatter(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    size_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = frame[size_col].clip(lower=1) * 0.8
    ax.scatter(frame[x_col], frame[y_col], s=sizes, alpha=0.7, color="#ff7f0e")
    for _, row in frame.sort_values(size_col, ascending=False).head(10).iterrows():
        ax.annotate(row[label_col], (row[x_col], row[y_col]), fontsize=8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_tool_mix(frame: pd.DataFrame, *, output_path: Path) -> Path:
    pivot = (
        frame.pivot_table(
            index="agent",
            columns="tool_category",
            values="share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    ordered_columns = [column for column in ["shell", "read", "edit", "plan", "other"] if column in pivot.columns]
    pivot = pivot[ordered_columns]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.55)))
    left = pd.Series(0.0, index=pivot.index)
    colors = {
        "shell": "#1f77b4",
        "read": "#2ca02c",
        "edit": "#d62728",
        "plan": "#9467bd",
        "other": "#8c564b",
    }
    for column in pivot.columns:
        ax.barh(
            pivot.index,
            pivot[column],
            left=left,
            label=column,
            color=colors.get(column, "#7f7f7f"),
        )
        left += pivot[column]
    ax.set_title("Tool Category Mix by Agent (runs with traces)")
    ax.set_xlabel("Share of Parsed Tool Calls")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_task_agent_heatmap(frame: pd.DataFrame, *, output_path: Path) -> Path:
    task_order = (
        frame.groupby("task_name")["success_rate"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    agent_order = (
        frame.groupby("agent")["runs"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    pivot = (
        frame.pivot_table(
            index="task_name",
            columns="agent",
            values="success_rate",
            aggfunc="mean",
        )
        .reindex(index=task_order, columns=agent_order)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(12, max(10, len(pivot) * 0.22)))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Task-by-Agent Success Heatmap")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="Success Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_focus_task_histogram(
    frame: pd.DataFrame,
    *,
    task_name: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    failed = frame[~frame["success"]]["tool_call_count"]
    succeeded = frame[frame["success"]]["tool_call_count"]
    bins = min(30, max(10, int(frame["tool_call_count"].max() // 3) if len(frame) else 10))
    ax.hist(failed, bins=bins, alpha=0.65, label="Unresolved", color="#d62728")
    ax.hist(succeeded, bins=bins, alpha=0.65, label="Resolved", color="#2ca02c")
    ax.set_title(f"Tool Call Distribution for {task_name}")
    ax.set_xlabel("Parsed Tool Calls")
    ax.set_ylabel("Runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_cols = result.select_dtypes(include=["number", "bool"]).columns
    for col in numeric_cols:
        if result[col].dtype == bool:
            continue
        result[col] = result[col].round(4)
    return result


def _empty_step_summary() -> dict[str, Any]:
    return {
        "step_count": 0,
        "agent_step_count": 0,
        "tool_call_count": 0,
        "shell_tool_calls": 0,
        "read_tool_calls": 0,
        "edit_tool_calls": 0,
        "plan_tool_calls": 0,
        "tool_counter": Counter(),
        "tool_category_counter": Counter(),
    }


def _has_steps(value: object) -> bool:
    return isinstance(value, str) and value not in {"", "null", "[]"}


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_success(value: object) -> bool | None:
    if value is None:
        return None
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return None


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")
