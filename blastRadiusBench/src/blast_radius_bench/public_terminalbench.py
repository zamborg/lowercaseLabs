"""Utilities for analyzing public Terminal-Bench trajectory datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import shlex
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
_SHELL_TOOL_NAMES = {"execute"}
_READ_TOOL_HINTS = ("read", "open", "cat", "grep")
_EDIT_TOOL_HINTS = ("edit", "write", "replace", "patch")
_PLAN_TOOL_HINTS = ("todo", "task_tracker", "think", "plan")
_TOOL_CATEGORIES = ("shell", "read", "edit", "plan", "other")
_COMMAND_INTENTS = (
    "inspect",
    "search",
    "read",
    "edit",
    "test",
    "build",
    "dependency",
    "run",
    "plan",
    "finish",
    "control",
    "other",
)


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

    run_df, agent_tool_df, agent_tool_category_df, agent_tool_intent_df = (
        _summarize_dataset(dataset, dataset_id)
    )

    agent_summary = _build_agent_summary(run_df)
    agent_summary = _add_intent_shares(agent_summary)
    model_summary = _build_model_summary(run_df, min_runs=max(10, min_runs // 4))
    task_summary = _build_task_summary(run_df)
    task_agent_summary = _build_task_agent_summary(run_df)
    agent_summary = _add_task_adjusted_success(
        agent_summary,
        task_agent_summary=task_agent_summary,
        task_summary=task_summary,
    )
    tool_summary = _build_tool_summary(agent_tool_df)
    intent_summary = _build_intent_summary(agent_tool_intent_df)
    agent_contrast_summary = _build_agent_contrast_summary(agent_summary)
    task_agent_contrast_summary = _build_task_agent_contrast_summary(task_agent_summary)
    same_model_agent_contrast_summary = _build_same_model_agent_contrast_summary(
        model_summary
    )
    same_agent_model_contrast_summary = _build_same_agent_model_contrast_summary(
        model_summary
    )
    process_correlation_summary = _build_process_correlation_summary(run_df)
    agent_task_centered_process_summary = _build_agent_task_centered_process_summary(
        run_df
    )
    agent_behavior_signature = _build_agent_behavior_signature(agent_summary)
    focus_task = _select_focus_task(task_summary)

    run_df.to_csv(output_path / "run_metrics.csv", index=False)
    agent_summary.to_csv(output_path / "agent_summary.csv", index=False)
    model_summary.to_csv(output_path / "model_summary.csv", index=False)
    task_summary.to_csv(output_path / "task_summary.csv", index=False)
    task_agent_summary.to_csv(output_path / "task_agent_summary.csv", index=False)
    tool_summary.to_csv(output_path / "tool_summary.csv", index=False)
    agent_tool_category_df.to_csv(output_path / "tool_category_summary.csv", index=False)
    agent_tool_intent_df.to_csv(output_path / "tool_intent_summary.csv", index=False)
    intent_summary.to_csv(output_path / "intent_summary.csv", index=False)
    agent_contrast_summary.to_csv(output_path / "agent_contrast_summary.csv", index=False)
    task_agent_contrast_summary.to_csv(
        output_path / "task_agent_contrast_summary.csv",
        index=False,
    )
    same_model_agent_contrast_summary.to_csv(
        output_path / "same_model_agent_contrast_summary.csv",
        index=False,
    )
    same_agent_model_contrast_summary.to_csv(
        output_path / "same_agent_model_contrast_summary.csv",
        index=False,
    )
    process_correlation_summary.to_csv(
        output_path / "process_correlation_summary.csv",
        index=False,
    )
    agent_task_centered_process_summary.to_csv(
        output_path / "agent_task_centered_process_summary.csv",
        index=False,
    )
    agent_behavior_signature.to_csv(
        output_path / "agent_behavior_signature.csv",
        index=False,
    )

    plot_paths = _write_plots(
        run_df=run_df,
        agent_summary=agent_summary,
        task_summary=task_summary,
        task_agent_summary=task_agent_summary,
        agent_tool_category_df=agent_tool_category_df,
        agent_tool_intent_df=agent_tool_intent_df,
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
        intent_summary=intent_summary,
        agent_contrast_summary=agent_contrast_summary,
        task_agent_contrast_summary=task_agent_contrast_summary,
        same_model_agent_contrast_summary=same_model_agent_contrast_summary,
        same_agent_model_contrast_summary=same_agent_model_contrast_summary,
        process_correlation_summary=process_correlation_summary,
        agent_task_centered_process_summary=agent_task_centered_process_summary,
        agent_behavior_signature=agent_behavior_signature,
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


def _summarize_dataset(
    dataset: Any,
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    agent_tool_counter: dict[str, Counter[str]] = defaultdict(Counter)
    agent_tool_category_counter: dict[str, Counter[str]] = defaultdict(Counter)
    agent_tool_intent_counter: dict[str, Counter[str]] = defaultdict(Counter)

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
                "other_tool_calls": step_summary["other_tool_calls"],
                "first_tool_step": step_summary["first_tool_step"],
                "first_shell_step": step_summary["first_shell_step"],
                "first_read_step": step_summary["first_read_step"],
                "first_edit_step": step_summary["first_edit_step"],
                "first_plan_step": step_summary["first_plan_step"],
                "pre_edit_tool_calls": step_summary["pre_edit_tool_calls"],
                "category_switch_rate": step_summary["category_switch_rate"],
                "longest_same_category_run": step_summary["longest_same_category_run"],
                "repeated_tool_command_rate": step_summary["repeated_tool_command_rate"],
                "tool_category_entropy": step_summary["tool_category_entropy"],
                "command_intent_entropy": step_summary["command_intent_entropy"],
                "intent_switch_rate": step_summary["intent_switch_rate"],
                "failed_tool_call_count": step_summary["failed_tool_call_count"],
                "failure_rate": step_summary["failure_rate"],
                "first_failure_step": step_summary["first_failure_step"],
                "post_failure_tool_calls": step_summary["post_failure_tool_calls"],
                "post_failure_intent_entropy": step_summary["post_failure_intent_entropy"],
                **{
                    f"{intent}_intent_calls": step_summary["intent_counter"][intent]
                    for intent in _COMMAND_INTENTS
                },
            }
        )

        if has_steps:
            agent_tool_counter[agent].update(step_summary["tool_counter"])
            agent_tool_category_counter[agent].update(step_summary["tool_category_counter"])
            agent_tool_intent_counter[agent].update(step_summary["intent_counter"])

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

    intent_rows = []
    for agent, counter in agent_tool_intent_counter.items():
        total = sum(counter.values())
        for intent, count in counter.items():
            intent_rows.append(
                {
                    "agent": agent,
                    "intent": intent,
                    "count": count,
                    "share": (count / total) if total else 0.0,
                }
            )
    agent_tool_intent_df = pd.DataFrame(intent_rows)
    if agent_tool_intent_df.empty:
        agent_tool_intent_df = pd.DataFrame(
            columns=["agent", "intent", "count", "share"]
        )

    return run_df, agent_tool_df, agent_tool_category_df, agent_tool_intent_df


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
    intent_counter: Counter[str] = Counter()
    command_counter: Counter[tuple[str, str]] = Counter()
    category_sequence: list[str] = []
    intent_sequence: list[str] = []
    tool_call_count = 0
    agent_step_count = 0
    first_tool_step: int | None = None
    first_category_step = {category: None for category in _TOOL_CATEGORIES}
    pre_edit_tool_calls = 0
    seen_edit = False
    failed_tool_call_count = 0
    first_failure_step: int | None = None
    seen_failure = False
    post_failure_intents: list[str] = []

    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("src") == "agent":
            agent_step_count += 1
        tools = step.get("tools") or []
        step_failed = _observation_failed(step.get("obs"))
        for tool in tools:
            tool_name = _public_tool_name(tool)
            if tool_name is None:
                continue
            category = classify_public_tool(tool_name)
            command = _public_tool_command(tool)
            intent = classify_public_command_intent(tool_name, command)
            tool_call_count += 1
            tool_counter[tool_name] += 1
            category_counter[category] += 1
            intent_counter[intent] += 1
            category_sequence.append(category)
            intent_sequence.append(intent)
            if seen_failure:
                post_failure_intents.append(intent)
            if step_failed:
                failed_tool_call_count += 1
                if first_failure_step is None:
                    first_failure_step = step_index
                seen_failure = True
            command_counter[(tool_name, command)] += 1
            if first_tool_step is None:
                first_tool_step = step_index
            if first_category_step[category] is None:
                first_category_step[category] = step_index
            if category == "edit":
                seen_edit = True
            elif not seen_edit:
                pre_edit_tool_calls += 1

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
        "other_tool_calls": sum(
            count for name, count in tool_counter.items() if classify_public_tool(name) == "other"
        ),
        "first_tool_step": first_tool_step,
        "first_shell_step": first_category_step["shell"],
        "first_read_step": first_category_step["read"],
        "first_edit_step": first_category_step["edit"],
        "first_plan_step": first_category_step["plan"],
        "pre_edit_tool_calls": pre_edit_tool_calls if seen_edit else None,
        "category_switch_rate": _category_switch_rate(category_sequence),
        "longest_same_category_run": _longest_same_category_run(category_sequence),
        "repeated_tool_command_rate": _repeated_tool_command_rate(command_counter),
        "tool_category_entropy": _tool_category_entropy(category_counter),
        "command_intent_entropy": _tool_category_entropy(intent_counter),
        "intent_switch_rate": _category_switch_rate(intent_sequence),
        "failed_tool_call_count": failed_tool_call_count,
        "failure_rate": _safe_ratio(failed_tool_call_count, tool_call_count),
        "first_failure_step": first_failure_step,
        "post_failure_tool_calls": len(post_failure_intents),
        "post_failure_intent_entropy": _tool_category_entropy(Counter(post_failure_intents)),
        "tool_counter": tool_counter,
        "tool_category_counter": category_counter,
        "intent_counter": intent_counter,
    }


def classify_public_tool(tool_name: str) -> str:
    """Map a public trajectory tool name to a coarse behavioral category."""
    lower_name = tool_name.lower()
    if lower_name in _SHELL_TOOL_NAMES or any(hint in lower_name for hint in _SHELL_TOOL_HINTS):
        return "shell"
    if any(hint in lower_name for hint in _PLAN_TOOL_HINTS):
        return "plan"
    if any(hint in lower_name for hint in _READ_TOOL_HINTS):
        return "read"
    if any(hint in lower_name for hint in _EDIT_TOOL_HINTS):
        return "edit"
    return "other"


def classify_public_command_intent(tool_name: str, command: str = "") -> str:
    """Map a tool call and optional shell command to a coarse action intent."""
    lower_tool_name = tool_name.lower()
    if lower_tool_name in {"finish", "mark_task_complete"}:
        return "finish"
    if "ipython" in lower_tool_name:
        return "run"

    category = classify_public_tool(tool_name)
    if category in {"plan", "read", "edit"}:
        return category

    lowered = _unwrap_shell_command(command).lower()
    if lowered.strip() in {"c-c", "ctrl-c", "^c"}:
        return "control"
    if "complete_task_and_submit_final_output" in lowered:
        return "finish"

    tokens = _shell_tokens(lowered)
    first = Path(tokens[0]).name if tokens else ""
    joined = " ".join(tokens)

    if _contains_command(joined, {"pytest", "bats", "jest", "rspec"}) or any(
        pattern in joined
        for pattern in (
            "npm test",
            "pnpm test",
            "yarn test",
            "go test",
            "cargo test",
            "mvn test",
            "gradle test",
            "make test",
            "python -m unittest",
            "python3 -m unittest",
        )
    ):
        return "test"

    if any(
        pattern in joined
        for pattern in (
            "apt-get install",
            "apt install",
            "pip install",
            "uv sync",
            "poetry install",
            "npm install",
            "pnpm install",
            "yarn install",
            "conda install",
        )
    ) or any(
        pattern in lowered
        for pattern in (
            "apt-get install",
            "apt install",
            "pip install",
            "uv sync",
            "poetry install",
            "npm install",
            "pnpm install",
            "yarn install",
            "conda install",
        )
    ) or first in {"wget", "curl", "opam"} or (first == "git" and "clone" in tokens):
        return "dependency"

    if first in {"rg", "grep", "find", "fd", "ack", "ag"}:
        return "search"
    if first in {"cat", "head", "tail", "less", "more", "nl"}:
        return "read"
    if first == "sed" and "-i" not in tokens:
        return "read"
    if first in {"ls", "pwd", "tree", "stat", "file", "du", "which", "wc"}:
        return "inspect"

    if (
        "apply_patch" in joined
        or "str_replace_editor" in joined
        or "sed -i" in joined
        or "perl -pi" in joined
        or "cat >" in lowered
        or "tee " in joined
        or first in {"chmod", "touch", "mv", "cp", "rm", "mkdir", "tar"}
    ):
        return "edit"

    if first in {"make", "cmake", "gcc", "g++", "clang", "clang++"} or any(
        pattern in joined
        for pattern in (
            "cargo build",
            "npm run build",
            "pnpm build",
            "yarn build",
            "docker build",
        )
    ):
        return "build"

    if first in {
        "python",
        "python3",
        "node",
        "ruby",
        "r",
        "rscript",
        "java",
        "sqlite3",
        "7z",
        "bash",
        "sh",
        "zsh",
    } or first.startswith(("./", "/")):
        return "run"

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


def _public_tool_command(tool: object) -> str:
    if not isinstance(tool, dict):
        return ""
    value = tool.get("cmd") or tool.get("command") or tool.get("input") or ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _unwrap_shell_command(command: str) -> str:
    value = str(command).strip()
    try:
        tokens = shlex.split(value)
    except ValueError:
        return value
    if len(tokens) >= 3 and Path(tokens[0]).name in {"bash", "sh", "zsh"}:
        for index, token in enumerate(tokens[1:], start=1):
            if token in {"-c", "-lc"} and index + 1 < len(tokens):
                return tokens[index + 1]
    return value


def _shell_tokens(command: str) -> list[str]:
    segments = command.replace("&&", "\n").replace(";", "\n").splitlines()
    for segment in segments:
        stripped = segment.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            tokens = stripped.split()
        if not tokens:
            continue
        first = Path(tokens[0]).name
        if first in {"cd", "export", "set"}:
            continue
        if first == "sudo" and len(tokens) > 1:
            return tokens[1:]
        return tokens
    return []


def _contains_command(command: str, names: set[str]) -> bool:
    padded = f" {command} "
    return any(f" {name} " in padded or padded.startswith(f" {name}") for name in names)


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
            mean_other_tool_calls=("other_tool_calls", "mean"),
            mean_first_edit_step=("first_edit_step", "mean"),
            mean_pre_edit_tool_calls=("pre_edit_tool_calls", "mean"),
            mean_category_switch_rate=("category_switch_rate", "mean"),
            mean_longest_same_category_run=("longest_same_category_run", "mean"),
            mean_repeated_tool_command_rate=("repeated_tool_command_rate", "mean"),
            mean_tool_category_entropy=("tool_category_entropy", "mean"),
            mean_command_intent_entropy=("command_intent_entropy", "mean"),
            mean_intent_switch_rate=("intent_switch_rate", "mean"),
            mean_failed_tool_call_count=("failed_tool_call_count", "mean"),
            mean_failure_rate=("failure_rate", "mean"),
            mean_first_failure_step=("first_failure_step", "mean"),
            mean_post_failure_tool_calls=("post_failure_tool_calls", "mean"),
            mean_post_failure_intent_entropy=("post_failure_intent_entropy", "mean"),
            mean_inspect_intent_calls=("inspect_intent_calls", "mean"),
            mean_search_intent_calls=("search_intent_calls", "mean"),
            mean_read_intent_calls=("read_intent_calls", "mean"),
            mean_edit_intent_calls=("edit_intent_calls", "mean"),
            mean_test_intent_calls=("test_intent_calls", "mean"),
            mean_build_intent_calls=("build_intent_calls", "mean"),
            mean_dependency_intent_calls=("dependency_intent_calls", "mean"),
            mean_run_intent_calls=("run_intent_calls", "mean"),
            mean_plan_intent_calls=("plan_intent_calls", "mean"),
            mean_other_intent_calls=("other_intent_calls", "mean"),
        )
        .reset_index()
        .sort_values(["runs", "success_rate"], ascending=[False, False])
    )
    summary = _add_success_failure_deltas(summary, run_df, group_cols=["agent"])
    summary = _add_tool_category_shares(summary)
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
            mean_tool_call_count=("tool_call_count", "mean"),
            mean_first_edit_step=("first_edit_step", "mean"),
            mean_pre_edit_tool_calls=("pre_edit_tool_calls", "mean"),
            mean_category_switch_rate=("category_switch_rate", "mean"),
            mean_repeated_tool_command_rate=("repeated_tool_command_rate", "mean"),
            mean_failure_rate=("failure_rate", "mean"),
            mean_post_failure_intent_entropy=("post_failure_intent_entropy", "mean"),
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


def _build_intent_summary(agent_tool_intent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_tool_intent_df.empty:
        return pd.DataFrame(columns=["intent", "count"])
    summary = (
        agent_tool_intent_df.groupby("intent", dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )
    total = summary["count"].sum()
    summary["share"] = summary["count"] / total if total else 0.0
    return _round_numeric(summary)


def _build_agent_contrast_summary(
    agent_summary: pd.DataFrame,
    *,
    min_runs: int = 400,
    min_trace_coverage: float = 0.3,
    max_success_gap: float = 0.05,
) -> pd.DataFrame:
    """Find agent pairs with similar outcomes but different process signatures."""
    eligible = agent_summary[
        (agent_summary["runs"] >= min_runs)
        & (agent_summary["trace_coverage"] >= min_trace_coverage)
        & (agent_summary["mean_tool_call_count"] > 0)
    ].copy()
    rows: list[dict[str, Any]] = []
    records = eligible.to_dict("records")
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            success_gap = abs(left["success_rate"] - right["success_rate"])
            if success_gap > max_success_gap:
                continue
            tool_call_gap = abs(
                left["mean_tool_call_count"] - right["mean_tool_call_count"]
            )
            switch_gap = abs(
                _zero_if_missing(left.get("mean_category_switch_rate"))
                - _zero_if_missing(right.get("mean_category_switch_rate"))
            )
            repeat_gap = abs(
                _zero_if_missing(left.get("mean_repeated_tool_command_rate"))
                - _zero_if_missing(right.get("mean_repeated_tool_command_rate"))
            )
            entropy_gap = abs(
                _zero_if_missing(left.get("mean_tool_category_entropy"))
                - _zero_if_missing(right.get("mean_tool_category_entropy"))
            )
            process_gap_score = (
                tool_call_gap
                + 25 * switch_gap
                + 25 * repeat_gap
                + 5 * entropy_gap
            )
            rows.append(
                {
                    "agent_a": left["agent"],
                    "agent_b": right["agent"],
                    "success_a": left["success_rate"],
                    "success_b": right["success_rate"],
                    "success_gap": success_gap,
                    "mean_tool_calls_a": left["mean_tool_call_count"],
                    "mean_tool_calls_b": right["mean_tool_call_count"],
                    "tool_call_gap": tool_call_gap,
                    "switch_rate_a": left.get("mean_category_switch_rate"),
                    "switch_rate_b": right.get("mean_category_switch_rate"),
                    "switch_rate_gap": switch_gap,
                    "repeat_rate_a": left.get("mean_repeated_tool_command_rate"),
                    "repeat_rate_b": right.get("mean_repeated_tool_command_rate"),
                    "repeat_rate_gap": repeat_gap,
                    "entropy_a": left.get("mean_tool_category_entropy"),
                    "entropy_b": right.get("mean_tool_category_entropy"),
                    "entropy_gap": entropy_gap,
                    "process_gap_score": process_gap_score,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "agent_a",
                "agent_b",
                "success_a",
                "success_b",
                "success_gap",
                "mean_tool_calls_a",
                "mean_tool_calls_b",
                "tool_call_gap",
                "switch_rate_gap",
                "repeat_rate_gap",
                "entropy_gap",
                "process_gap_score",
            ]
        )
    return _round_numeric(
        pd.DataFrame(rows).sort_values("process_gap_score", ascending=False)
    )


def _build_process_correlation_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    traced = run_df[run_df["has_steps"]].copy()
    if traced.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "n",
                "raw_success_corr",
                "task_centered_success_corr",
                "feature_mean",
            ]
        )

    traced["success_float"] = traced["success"].astype(float)
    task_success = traced.groupby("task_name")["success_float"].transform("mean")
    traced["task_centered_success"] = traced["success_float"] - task_success

    feature_cols = [
        "tool_call_count",
        "step_count",
        "category_switch_rate",
        "intent_switch_rate",
        "repeated_tool_command_rate",
        "tool_category_entropy",
        "command_intent_entropy",
        "inspect_intent_calls",
        "search_intent_calls",
        "read_intent_calls",
        "edit_intent_calls",
        "test_intent_calls",
        "build_intent_calls",
        "dependency_intent_calls",
        "run_intent_calls",
        "failed_tool_call_count",
        "failure_rate",
        "post_failure_tool_calls",
        "post_failure_intent_entropy",
        "duration_seconds",
        "cost_cents",
        "input_tokens",
    ]
    rows = []
    for feature in feature_cols:
        if feature not in traced.columns:
            continue
        feature_series = pd.to_numeric(traced[feature], errors="coerce")
        valid = feature_series.notna()
        if int(valid.sum()) < 30:
            continue
        task_feature = feature_series.groupby(traced["task_name"]).transform("mean")
        centered_feature = feature_series - task_feature
        rows.append(
            {
                "feature": feature,
                "n": int(valid.sum()),
                "raw_success_corr": _corr(feature_series[valid], traced.loc[valid, "success_float"]),
                "task_centered_success_corr": _corr(
                    centered_feature[valid],
                    traced.loc[valid, "task_centered_success"],
                ),
                "feature_mean": feature_series[valid].mean(),
            }
        )
    return _round_numeric(
        pd.DataFrame(rows).sort_values(
            "task_centered_success_corr",
            ascending=False,
            na_position="last",
        )
    )


def _build_agent_task_centered_process_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    traced = run_df[run_df["has_steps"]].copy()
    if traced.empty:
        return pd.DataFrame()
    features = [
        "success",
        "tool_call_count",
        "command_intent_entropy",
        "tool_category_entropy",
        "category_switch_rate",
        "repeated_tool_command_rate",
        "failure_rate",
        "post_failure_intent_entropy",
        "duration_seconds",
        "cost_cents",
    ]
    traced["success"] = traced["success"].astype(float)
    centered_cols = []
    for feature in features:
        if feature not in traced.columns:
            continue
        values = pd.to_numeric(traced[feature], errors="coerce")
        task_mean = values.groupby(traced["task_name"]).transform("mean")
        centered_col = f"task_centered_{feature}"
        traced[centered_col] = values - task_mean
        centered_cols.append(centered_col)

    agg_spec = {
        "runs": ("agent", "size"),
        "tasks": ("task_name", "nunique"),
        "trace_coverage_within_traced_subset": ("has_steps", "mean"),
    }
    for column in centered_cols:
        agg_spec[f"mean_{column}"] = (column, "mean")
    summary = traced.groupby("agent", dropna=False).agg(**agg_spec).reset_index()
    sort_col = "mean_task_centered_success"
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False)
    return _round_numeric(summary)


def _build_agent_behavior_signature(agent_summary: pd.DataFrame) -> pd.DataFrame:
    if agent_summary.empty:
        return pd.DataFrame()
    eligible = agent_summary[
        (agent_summary["runs"] >= 100)
        & (agent_summary["trace_coverage"] >= 0.3)
        & (agent_summary["mean_tool_call_count"] > 0)
    ].copy()
    if eligible.empty:
        return pd.DataFrame()

    feature_cols = [
        "success_rate",
        "task_adjusted_success_lift",
        "mean_tool_call_count",
        "mean_category_switch_rate",
        "mean_intent_switch_rate",
        "mean_repeated_tool_command_rate",
        "mean_failure_rate",
        "mean_post_failure_intent_entropy",
        "mean_tool_category_entropy",
        "mean_command_intent_entropy",
        "mean_shell_tool_share",
        "mean_inspect_intent_share",
        "mean_search_intent_share",
        "mean_read_intent_share",
        "mean_edit_intent_share",
        "mean_test_intent_share",
        "mean_run_intent_share",
    ]
    output_cols = [
        "agent",
        "runs",
        "trace_coverage",
        *[col for col in feature_cols if col in eligible.columns],
    ]
    result = eligible[output_cols].copy()
    for column in feature_cols:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        std = values.std()
        if std is None or pd.isna(std) or std == 0:
            result[f"z_{column}"] = 0.0
        else:
            result[f"z_{column}"] = (values - values.mean()) / std

    tool_q25 = result["mean_tool_call_count"].quantile(0.25)
    repeat_q75 = result["mean_repeated_tool_command_rate"].quantile(0.75)
    switch_q75 = result["mean_category_switch_rate"].quantile(0.75)
    shell_q75 = result["mean_shell_tool_share"].quantile(0.75)

    result["signature_label"] = [
        _label_signature(row, tool_q25, repeat_q75, switch_q75, shell_q75)
        for _, row in result.iterrows()
    ]
    return _round_numeric(
        result.sort_values(
            ["task_adjusted_success_lift", "success_rate"],
            ascending=[False, False],
        )
    )


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
            mean_first_edit_step=("first_edit_step", "mean"),
            mean_pre_edit_tool_calls=("pre_edit_tool_calls", "mean"),
            mean_category_switch_rate=("category_switch_rate", "mean"),
            mean_repeated_tool_command_rate=("repeated_tool_command_rate", "mean"),
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
            mean_first_edit_step=("first_edit_step", "mean"),
            mean_pre_edit_tool_calls=("pre_edit_tool_calls", "mean"),
            mean_category_switch_rate=("category_switch_rate", "mean"),
            mean_repeated_tool_command_rate=("repeated_tool_command_rate", "mean"),
            mean_failure_rate=("failure_rate", "mean"),
            mean_post_failure_intent_entropy=("post_failure_intent_entropy", "mean"),
            mean_command_intent_entropy=("command_intent_entropy", "mean"),
            mean_intent_switch_rate=("intent_switch_rate", "mean"),
        )
        .reset_index()
    )
    return _round_numeric(summary)


def _build_task_agent_contrast_summary(
    task_agent_summary: pd.DataFrame,
    *,
    min_runs_per_cell: int = 3,
    max_success_gap: float = 0.1,
) -> pd.DataFrame:
    eligible = task_agent_summary[
        (task_agent_summary["runs"] >= min_runs_per_cell)
        & (task_agent_summary["trace_coverage"] > 0)
        & (task_agent_summary["mean_tool_call_count"] > 0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for task_name, group in eligible.groupby("task_name", dropna=False):
        records = group.sort_values("agent").to_dict("records")
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                success_gap = abs(left["success_rate"] - right["success_rate"])
                if success_gap > max_success_gap:
                    continue
                tool_call_gap = abs(
                    left["mean_tool_call_count"] - right["mean_tool_call_count"]
                )
                switch_gap = abs(
                    _zero_if_missing(left.get("mean_category_switch_rate"))
                    - _zero_if_missing(right.get("mean_category_switch_rate"))
                )
                repeat_gap = abs(
                    _zero_if_missing(left.get("mean_repeated_tool_command_rate"))
                    - _zero_if_missing(right.get("mean_repeated_tool_command_rate"))
                )
                intent_entropy_gap = abs(
                    _zero_if_missing(left.get("mean_command_intent_entropy"))
                    - _zero_if_missing(right.get("mean_command_intent_entropy"))
                )
                process_gap_score = (
                    tool_call_gap
                    + 25 * switch_gap
                    + 25 * repeat_gap
                    + 5 * intent_entropy_gap
                )
                rows.append(
                    {
                        "task_name": task_name,
                        "agent_a": left["agent"],
                        "agent_b": right["agent"],
                        "runs_a": left["runs"],
                        "runs_b": right["runs"],
                        "success_a": left["success_rate"],
                        "success_b": right["success_rate"],
                        "success_gap": success_gap,
                        "mean_tool_calls_a": left["mean_tool_call_count"],
                        "mean_tool_calls_b": right["mean_tool_call_count"],
                        "tool_call_gap": tool_call_gap,
                        "switch_rate_gap": switch_gap,
                        "repeat_rate_gap": repeat_gap,
                        "intent_entropy_gap": intent_entropy_gap,
                        "process_gap_score": process_gap_score,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "task_name",
                "agent_a",
                "agent_b",
                "success_gap",
                "tool_call_gap",
                "switch_rate_gap",
                "repeat_rate_gap",
                "intent_entropy_gap",
                "process_gap_score",
            ]
        )
    return _round_numeric(
        pd.DataFrame(rows).sort_values("process_gap_score", ascending=False)
    )


def _build_same_model_agent_contrast_summary(
    model_summary: pd.DataFrame,
    *,
    min_runs_per_cell: int = 100,
    min_trace_coverage: float = 0.3,
) -> pd.DataFrame:
    return _build_model_agent_contrasts(
        model_summary,
        group_col="model",
        item_col="agent",
        group_label="model",
        item_label="agent",
        min_runs_per_cell=min_runs_per_cell,
        min_trace_coverage=min_trace_coverage,
    )


def _build_same_agent_model_contrast_summary(
    model_summary: pd.DataFrame,
    *,
    min_runs_per_cell: int = 100,
    min_trace_coverage: float = 0.3,
) -> pd.DataFrame:
    return _build_model_agent_contrasts(
        model_summary,
        group_col="agent",
        item_col="model",
        group_label="agent",
        item_label="model",
        min_runs_per_cell=min_runs_per_cell,
        min_trace_coverage=min_trace_coverage,
    )


def _build_model_agent_contrasts(
    model_summary: pd.DataFrame,
    *,
    group_col: str,
    item_col: str,
    group_label: str,
    item_label: str,
    min_runs_per_cell: int,
    min_trace_coverage: float,
) -> pd.DataFrame:
    eligible = model_summary[
        (model_summary["runs"] >= min_runs_per_cell)
        & (model_summary["trace_coverage"] >= min_trace_coverage)
        & (model_summary["mean_tool_call_count"] > 0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for group_value, group in eligible.groupby(group_col, dropna=False):
        records = group.sort_values(item_col).to_dict("records")
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                tool_call_gap = abs(
                    left["mean_tool_call_count"] - right["mean_tool_call_count"]
                )
                switch_gap = abs(
                    _zero_if_missing(left.get("mean_category_switch_rate"))
                    - _zero_if_missing(right.get("mean_category_switch_rate"))
                )
                repeat_gap = abs(
                    _zero_if_missing(left.get("mean_repeated_tool_command_rate"))
                    - _zero_if_missing(right.get("mean_repeated_tool_command_rate"))
                )
                process_gap_score = tool_call_gap + 25 * switch_gap + 25 * repeat_gap
                rows.append(
                    {
                        group_label: group_value,
                        f"{item_label}_a": left[item_col],
                        f"{item_label}_b": right[item_col],
                        "runs_a": left["runs"],
                        "runs_b": right["runs"],
                        "success_a": left["success_rate"],
                        "success_b": right["success_rate"],
                        "success_gap": abs(left["success_rate"] - right["success_rate"]),
                        "mean_tool_calls_a": left["mean_tool_call_count"],
                        "mean_tool_calls_b": right["mean_tool_call_count"],
                        "tool_call_gap": tool_call_gap,
                        "switch_rate_gap": switch_gap,
                        "repeat_rate_gap": repeat_gap,
                        "process_gap_score": process_gap_score,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                group_label,
                f"{item_label}_a",
                f"{item_label}_b",
                "success_gap",
                "tool_call_gap",
                "switch_rate_gap",
                "repeat_rate_gap",
                "process_gap_score",
            ]
        )
    return _round_numeric(
        pd.DataFrame(rows).sort_values("process_gap_score", ascending=False)
    )


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
    agent_tool_intent_df: pd.DataFrame,
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
        if "mean_command_intent_entropy" in eligible.columns:
            plot_paths.append(
                _plot_scatter(
                    eligible.dropna(subset=["mean_command_intent_entropy"]),
                    x_col="mean_command_intent_entropy",
                    y_col="task_adjusted_success_lift",
                    size_col="runs",
                    label_col="agent",
                    title=f"Task-Adjusted Success vs Intent Entropy (min {min_runs} runs)",
                    xlabel="Mean Command Intent Entropy",
                    ylabel="Task-Adjusted Success Lift",
                    output_path=output_dir / "agent_success_lift_vs_intent_entropy.png",
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

    if not agent_tool_intent_df.empty:
        traced_counts = (
            run_df.groupby("agent", dropna=False)["has_steps"].sum().reset_index(name="traced_runs")
        )
        top_traced_agents = traced_counts.sort_values("traced_runs", ascending=False).head(8)["agent"]
        intent_mix_df = agent_tool_intent_df[
            agent_tool_intent_df["agent"].isin(top_traced_agents)
        ].copy()
        if not intent_mix_df.empty:
            plot_paths.append(
                _plot_intent_mix(
                    intent_mix_df,
                    output_path=output_dir / "agent_intent_mix.png",
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
    intent_summary: pd.DataFrame,
    agent_contrast_summary: pd.DataFrame,
    task_agent_contrast_summary: pd.DataFrame,
    same_model_agent_contrast_summary: pd.DataFrame,
    same_agent_model_contrast_summary: pd.DataFrame,
    process_correlation_summary: pd.DataFrame,
    agent_task_centered_process_summary: pd.DataFrame,
    agent_behavior_signature: pd.DataFrame,
    plot_paths: list[Path],
    min_runs: int,
    focus_task: str | None,
) -> Path:
    dataset_url = PUBLIC_TERMINALBENCH_DATASETS.get(dataset_id, "")
    output_path = output_dir / "index.html"
    traced_runs = int(run_df["has_steps"].sum())
    same_task_contrasts = len(task_agent_contrast_summary)
    top_process_gap = (
        float(task_agent_contrast_summary["process_gap_score"].max())
        if not task_agent_contrast_summary.empty
        else 0.0
    )
    other_intent_share = _lookup_share(intent_summary, "intent", "other")

    headline_stats = {
        "Runs": f"{len(run_df):,}",
        "Traced runs": f"{traced_runs:,}",
        "Agents": f"{run_df['agent'].nunique():,}",
        "Models": f"{run_df['model'].nunique():,}",
        "Tasks": f"{run_df['task_name'].nunique():,}",
        "Success rate": f"{run_df['success'].mean():.1%}",
        "Trace coverage": f"{run_df['has_steps'].mean():.1%}",
        "Same-task contrasts": f"{same_task_contrasts:,}",
        "Max process gap": f"{top_process_gap:.1f}",
        "Other intent share": f"{other_intent_share:.1%}",
    }

    stats_html = "".join(
        f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in headline_stats.items()
    )
    plot_lookup = {path.stem: path.name for path in plot_paths}
    signature_cards = _signature_cards_html(agent_behavior_signature)
    plot_grid = _plot_grid_html(
        plot_lookup,
        [
            ("agent_success_lift_vs_intent_entropy", "Success Lift vs Intent Entropy"),
            ("agent_intent_mix", "Agent Intent Mix"),
            ("agent_tool_mix", "Agent Tool Mix"),
            ("task_agent_success_heatmap", "Task-Agent Success Heatmap"),
        ],
    )
    evidence_table = _dashboard_table(
        task_agent_contrast_summary,
        ["task_name", "agent_a", "agent_b", "success_gap", "tool_call_gap", "process_gap_score"],
        limit=16,
    )
    same_model_table = _dashboard_table(
        same_model_agent_contrast_summary,
        ["model", "agent_a", "agent_b", "success_gap", "tool_call_gap", "process_gap_score"],
        limit=14,
    )
    residual_table = _dashboard_table(
        agent_task_centered_process_summary,
        [
            "agent",
            "mean_task_centered_success",
            "mean_task_centered_tool_call_count",
            "mean_task_centered_command_intent_entropy",
            "mean_task_centered_failure_rate",
            "mean_task_centered_duration_seconds",
        ],
        limit=20,
    )
    correlation_table = _dashboard_table(
        process_correlation_summary,
        ["feature", "raw_success_corr", "task_centered_success_corr", "feature_mean"],
        limit=24,
    )
    intent_table = _dashboard_table(intent_summary, ["intent", "count", "share"], limit=20)
    agent_table = _dashboard_table(
        agent_summary,
        [
            "agent",
            "runs",
            "success_rate",
            "task_adjusted_success_lift",
            "trace_coverage",
            "mean_tool_call_count",
            "mean_command_intent_entropy",
            "mean_failure_rate",
        ],
        limit=30,
        table_id="agent-summary-table",
    )
    model_table = _dashboard_table(
        model_summary,
        [
            "agent",
            "model",
            "runs",
            "success_rate",
            "trace_coverage",
            "mean_tool_call_count",
            "mean_category_switch_rate",
        ],
        limit=40,
        table_id="model-summary-table",
    )
    task_table = _dashboard_table(
        task_summary.sort_values(["success_rate", "runs"], ascending=[True, False]),
        ["task_name", "runs", "success_rate", "trace_coverage", "mean_tool_call_count", "mean_failure_rate"],
        limit=40,
        table_id="task-summary-table",
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>blastRadiusBench Dashboard</title>
  <style>
    :root {{
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #62625f;
      --line: #dedbd2;
      --green: #1d7f64;
      --red: #b13f31;
      --gold: #b67b1f;
      --blue: #2e6f9e;
      --violet: #7856a8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); line-height: 1.45; }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    header {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr); gap: 24px; align-items: end; padding: 28px 0 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: clamp(32px, 5vw, 72px); line-height: 0.95; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    a {{ color: var(--blue); }}
    code {{ background: #ece9df; padding: 2px 6px; border-radius: 4px; }}
    .lede {{ max-width: 860px; margin-top: 14px; font-size: 18px; color: #343431; }}
    .meta {{ display: grid; gap: 8px; align-content: end; font-size: 14px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 20px 0 28px; }}
    .metric {{ min-height: 82px; border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px 14px; display: grid; align-content: space-between; }}
    .metric span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric strong {{ font-size: 26px; line-height: 1; }}
    .band {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .thesis {{ display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: center; border-left: 6px solid var(--green); }}
    .thesis strong {{ font-size: clamp(24px, 3vw, 42px); line-height: 1; }}
    .pillrow {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: 12px; background: #fbfaf6; color: #353530; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .signature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .sig {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; display: grid; gap: 10px; }}
    .sig-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }}
    .tag {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; border-radius: 999px; padding: 4px 8px; background: #eee8da; }}
    .tag.compact {{ background: #dfeee8; color: #145c48; }}
    .tag.repetitive {{ background: #f4ded9; color: #873026; }}
    .tag.shell-linear {{ background: #e6e0f0; color: #563a7d; }}
    .tag.multi-tool {{ background: #dfeaf2; color: #1f5578; }}
    .tag.mixed {{ background: #f3e8cc; color: #7d5618; }}
    .bars {{ display: grid; gap: 7px; }}
    .barline {{ display: grid; grid-template-columns: 90px 1fr 52px; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }}
    .track {{ height: 7px; border-radius: 999px; background: #ece9df; overflow: hidden; }}
    .fill {{ height: 100%; background: var(--green); border-radius: inherit; }}
    .fill.red {{ background: var(--red); }}
    .fill.gold {{ background: var(--gold); }}
    .fill.blue {{ background: var(--blue); }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .plot-card {{ border: 1px solid var(--line); background: #fff; border-radius: 8px; padding: 12px; }}
    .plot-card img {{ width: 100%; display: block; border-radius: 6px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #ece9df; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f4f1e8; z-index: 1; color: #3a3935; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
    tr:hover td {{ background: #fbfaf6; }}
    .controls {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .filter {{ width: min(420px, 100%); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 12px; }}
    .section-head p {{ max-width: 760px; }}
    @media (max-width: 980px) {{ .shell {{ padding: 16px; }} header, .grid-2, .plot-grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>blastRadiusBench</h1>
        <p class="lede">A process dashboard for coding agents: not just whether they solve Terminal-Bench tasks, but how each model-harness tuple moves through the terminal world.</p>
      </div>
      <div class="meta">
        <div>Dataset <code>{dataset_id}</code></div>
        <div>Source <a href="{dataset_url}">{dataset_url}</a></div>
        <div>Focus task <code>{focus_task or "n/a"}</code></div>
        <div>Aggregate threshold <code>{min_runs}</code> runs</div>
      </div>
    </header>

    <section class="metrics">{stats_html}</section>

    <section class="band thesis">
      <div>
        <strong>Success-equivalent agents are not behavior-equivalent.</strong>
        <p class="lede">The strongest evidence is same-task contrast: agents can have nearly identical outcomes while differing by 100+ tool calls and sharply different action rhythms.</p>
      </div>
      <div class="pillrow">
        <span class="pill">Task-centered residuals</span>
        <span class="pill">Intent entropy</span>
        <span class="pill">Failure recovery</span>
        <span class="pill">Harness effects</span>
      </div>
    </section>

    <section class="band">
      <div class="section-head"><div><h2>Agent Behavior Signatures</h2><p>Task-adjusted success, process volume, failure rate, and command-intent diversity summarized by traced agent.</p></div></div>
      {signature_cards}
    </section>

    <section class="grid-2">
      <div class="band"><div class="section-head"><div><h2>Best Same-Task Evidence</h2><p>Same task, similar local outcome, divergent process vector.</p></div></div>{evidence_table}</div>
      <div class="band"><div class="section-head"><div><h2>Same Model, Different Harness</h2><p>Holding the model fixed still leaves large process differences.</p></div></div>{same_model_table}</div>
    </section>

    <section class="band"><div class="section-head"><div><h2>Visual Readout</h2><p>Charts generated from the public trace corpus and refreshed with the CSV artifacts.</p></div></div>{plot_grid}</section>

    <section class="grid-2">
      <div class="band"><h2>Task-Centered Agent Residuals</h2>{residual_table}</div>
      <div class="band"><h2>Process Correlations</h2>{correlation_table}</div>
    </section>

    <section class="grid-2">
      <div class="band"><h2>Command Intent Mix</h2>{intent_table}</div>
      <div class="band"><h2>Agent Summary Explorer</h2><div class="controls"><input class="filter" data-target="agent-summary-table" placeholder="Filter agents, signatures, values" /></div>{agent_table}</div>
    </section>

    <section class="grid-2">
      <div class="band"><h2>Model Summary Explorer</h2><div class="controls"><input class="filter" data-target="model-summary-table" placeholder="Filter models or agents" /></div>{model_table}</div>
      <div class="band"><h2>Hard Task Explorer</h2><div class="controls"><input class="filter" data-target="task-summary-table" placeholder="Filter task names" /></div>{task_table}</div>
    </section>
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
    output_path.write_text(html)
    return output_path


def _lookup_share(frame: pd.DataFrame, key_col: str, key: str) -> float:
    if frame.empty or key_col not in frame.columns or "share" not in frame.columns:
        return 0.0
    rows = frame[frame[key_col] == key]
    if rows.empty:
        return 0.0
    return float(rows.iloc[0]["share"])


def _signature_cards_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No signature data available.</p>"
    cards = []
    for _, row in frame.iterrows():
        label = str(row.get("signature_label", "mixed"))
        label_class = html.escape(label)
        agent = html.escape(str(row.get("agent", "unknown")))
        success = _coerce_display_float(row.get("success_rate"))
        lift = _coerce_display_float(row.get("task_adjusted_success_lift"))
        tool_calls = _coerce_display_float(row.get("mean_tool_call_count"))
        entropy = _coerce_display_float(row.get("mean_command_intent_entropy"))
        failure = _coerce_display_float(row.get("mean_failure_rate"))
        cards.append(
            "<article class='sig'>"
            "<div class='sig-top'>"
            f"<div><h3>{agent}</h3><p>{_format_pct(success)} success / {_format_signed(lift)} lift</p></div>"
            f"<span class='tag {label_class}'>{html.escape(label)}</span>"
            "</div>"
            "<div class='bars'>"
            f"{_barline('Success', success, _format_pct(success), 'green')}"
            f"{_barline('Tool calls', min(tool_calls / 80, 1), f'{tool_calls:.1f}', 'blue')}"
            f"{_barline('Entropy', min(entropy / 2.4, 1), f'{entropy:.2f}', 'gold')}"
            f"{_barline('Failure', failure, _format_pct(failure), 'red')}"
            "</div>"
            "</article>"
        )
    return f"<div class='signature-grid'>{''.join(cards)}</div>"


def _barline(label: str, ratio: float, value: str, color: str) -> str:
    width = max(0.0, min(float(ratio or 0.0), 1.0)) * 100
    return (
        "<div class='barline'>"
        f"<span>{html.escape(label)}</span>"
        f"<div class='track'><div class='fill {color}' style='width: {width:.1f}%'></div></div>"
        f"<span>{html.escape(value)}</span>"
        "</div>"
    )


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


def _coerce_display_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


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
    if frame[y_col].min() >= 0 and frame[y_col].max() <= 1:
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


def _plot_intent_mix(frame: pd.DataFrame, *, output_path: Path) -> Path:
    pivot = (
        frame.pivot_table(
            index="agent",
            columns="intent",
            values="share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    ordered_columns = [column for column in _COMMAND_INTENTS if column in pivot.columns]
    pivot = pivot[ordered_columns]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.55)))
    left = pd.Series(0.0, index=pivot.index)
    colors = {
        "inspect": "#1f77b4",
        "search": "#17becf",
        "read": "#2ca02c",
        "edit": "#d62728",
        "test": "#bcbd22",
        "build": "#ff7f0e",
        "dependency": "#8c564b",
        "run": "#e377c2",
        "plan": "#9467bd",
        "other": "#7f7f7f",
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
    ax.set_title("Command Intent Mix by Agent (runs with traces)")
    ax.set_xlabel("Share of Parsed Tool Calls")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
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


def _add_success_failure_deltas(
    summary: pd.DataFrame,
    run_df: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    traced = run_df[run_df["has_steps"]].copy()
    if traced.empty:
        return summary
    grouped = (
        traced.groupby(group_cols + ["success"], dropna=False)
        .agg(
            conditional_mean_tool_call_count=("tool_call_count", "mean"),
            conditional_mean_step_count=("step_count", "mean"),
            conditional_mean_category_switch_rate=("category_switch_rate", "mean"),
        )
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=group_cols,
        columns="success",
        values=[
            "conditional_mean_tool_call_count",
            "conditional_mean_step_count",
            "conditional_mean_category_switch_rate",
        ],
        aggfunc="first",
    )
    pivot.columns = [
        f"{metric}_{'success' if success else 'failure'}"
        for metric, success in pivot.columns
    ]
    pivot = pivot.reset_index()
    merged = summary.merge(pivot, on=group_cols, how="left")
    success_col = "conditional_mean_tool_call_count_success"
    failure_col = "conditional_mean_tool_call_count_failure"
    if success_col in merged.columns and failure_col in merged.columns:
        merged["tool_call_success_minus_failure"] = (
            merged[success_col] - merged[failure_col]
        )
    return merged


def _add_task_adjusted_success(
    agent_summary: pd.DataFrame,
    *,
    task_agent_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
) -> pd.DataFrame:
    if task_agent_summary.empty or task_summary.empty:
        return agent_summary
    task_baseline = task_summary[["task_name", "success_rate"]].rename(
        columns={"success_rate": "task_success_rate"}
    )
    merged = task_agent_summary.merge(task_baseline, on="task_name", how="left")
    merged["success_lift_vs_task"] = (
        merged["success_rate"] - merged["task_success_rate"]
    )
    weighted_rows = []
    for agent, group in merged.groupby("agent", dropna=False):
        total_runs = group["runs"].sum()
        if total_runs == 0:
            weighted_lift = None
        else:
            weighted_lift = (
                group["success_lift_vs_task"] * group["runs"]
            ).sum() / total_runs
        weighted_rows.append(
            {
                "agent": agent,
                "task_adjusted_success_lift": weighted_lift,
                "task_agent_cells": len(group),
            }
        )
    return agent_summary.merge(pd.DataFrame(weighted_rows), on="agent", how="left")


def _add_tool_category_shares(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    denominator = result["mean_tool_call_count"].replace(0, pd.NA)
    for category in ("shell", "read", "edit", "other"):
        count_col = f"mean_{category}_tool_calls"
        if count_col in result.columns:
            result[f"mean_{category}_tool_share"] = result[count_col] / denominator
    return result


def _add_intent_shares(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    denominator = result["mean_tool_call_count"].replace(0, pd.NA)
    for intent in _COMMAND_INTENTS:
        count_col = f"mean_{intent}_intent_calls"
        if count_col in result.columns:
            result[f"mean_{intent}_intent_share"] = result[count_col] / denominator
    return result


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 2:
        return None
    if frame.iloc[:, 0].std() == 0 or frame.iloc[:, 1].std() == 0:
        return None
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _label_signature(
    row: pd.Series,
    tool_q25: float,
    repeat_q75: float,
    switch_q75: float,
    shell_q75: float,
) -> str:
    if row["mean_tool_call_count"] <= tool_q25:
        return "compact"
    if row["mean_repeated_tool_command_rate"] >= repeat_q75:
        return "repetitive"
    if row["mean_category_switch_rate"] >= switch_q75:
        return "multi-tool"
    if row["mean_shell_tool_share"] >= shell_q75:
        return "shell-linear"
    return "mixed"


def _empty_step_summary() -> dict[str, Any]:
    return {
        "step_count": 0,
        "agent_step_count": 0,
        "tool_call_count": 0,
        "shell_tool_calls": 0,
        "read_tool_calls": 0,
        "edit_tool_calls": 0,
        "plan_tool_calls": 0,
        "other_tool_calls": 0,
        "first_tool_step": None,
        "first_shell_step": None,
        "first_read_step": None,
        "first_edit_step": None,
        "first_plan_step": None,
        "pre_edit_tool_calls": None,
        "category_switch_rate": None,
        "longest_same_category_run": 0,
        "repeated_tool_command_rate": None,
        "tool_category_entropy": None,
        "command_intent_entropy": None,
        "intent_switch_rate": None,
        "failed_tool_call_count": 0,
        "failure_rate": None,
        "first_failure_step": None,
        "post_failure_tool_calls": 0,
        "post_failure_intent_entropy": None,
        "tool_counter": Counter(),
        "tool_category_counter": Counter(),
        "intent_counter": Counter(),
    }


def _category_switch_rate(categories: list[str]) -> float | None:
    if len(categories) < 2:
        return None
    switches = sum(
        1
        for previous, current in zip(categories, categories[1:], strict=False)
        if previous != current
    )
    return switches / (len(categories) - 1)


def _longest_same_category_run(categories: list[str]) -> int:
    if not categories:
        return 0
    longest = 1
    current = 1
    for previous, category in zip(categories, categories[1:], strict=False):
        if previous == category:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def _repeated_tool_command_rate(counter: Counter[tuple[str, str]]) -> float | None:
    total = sum(counter.values())
    if total == 0:
        return None
    repeated = sum(count - 1 for count in counter.values() if count > 1)
    return repeated / total


def _tool_category_entropy(counter: Counter[str]) -> float | None:
    total = sum(counter.values())
    if total == 0:
        return None
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _observation_failed(observation: object) -> bool:
    if not isinstance(observation, str) or not observation.strip():
        return False
    lowered = observation.lower()
    failure_markers = (
        "command failed",
        "exit code 1",
        "exit code 2",
        "exit code 127",
        "traceback",
        "error:",
        "not found",
        "no such file",
        "permission denied",
        "failed with",
    )
    return any(marker in lowered for marker in failure_markers)


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


def _zero_if_missing(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
