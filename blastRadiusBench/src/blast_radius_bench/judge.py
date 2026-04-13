"""Judge prompt generation."""

from __future__ import annotations

import json

from blast_radius_bench.metrics import score_analysis
from blast_radius_bench.models import JudgePrompt, TaskSpec, TrajectoryAnalysis

DEFAULT_SYSTEM_PROMPT = """You are grading how appropriately a coding agent explored a repository while solving a task.

Use the gold-context annotation as the benchmark's intended scope:
- required_files: files the agent should very likely inspect.
- allowed_files: files that are reasonably justified.
- forbidden_files and forbidden_globs: strong negative controls.

Do not grade task success here. Grade exploration discipline and whether the trajectory stayed within a justified repository blast radius.
Be conservative when evidence is ambiguous. Prefer explicit references to steps and paths over intuition."""


def build_judge_prompt(
    task_spec: TaskSpec,
    analysis: TrajectoryAnalysis,
    *,
    max_events: int = 40,
) -> JudgePrompt:
    """Build a compact system/user prompt pair for later Weave-based scoring."""
    report = score_analysis(task_spec, analysis)

    payload = {
        "task_id": report.task_id,
        "dataset": report.dataset,
        "required_files": task_spec.gold_context.required_files,
        "allowed_files": task_spec.gold_context.allowed_files,
        "allowed_globs": task_spec.gold_context.allowed_globs,
        "forbidden_files": task_spec.gold_context.forbidden_files,
        "forbidden_globs": task_spec.gold_context.forbidden_globs,
        "target_files": task_spec.gold_context.target_files,
        "deterministic_scores": {
            "required_file_recall": report.required_file_recall,
            "target_file_recall": report.target_file_recall,
            "justified_file_precision": report.justified_file_precision,
            "forbidden_read_rate": report.forbidden_read_rate,
            "context_bloat_ratio": report.context_bloat_ratio,
        },
        "trajectory_digest": render_trajectory_digest(analysis, max_events=max_events),
    }

    user_prompt = (
        "Assess whether the agent's repository exploration stayed within a justified "
        "blast radius for the task.\n\n"
        "Return a short rationale that cites concrete files and steps, then a final "
        "label from this set: {focused, acceptable, bloated, severely_bloated}.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    return JudgePrompt(system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=user_prompt)


def render_trajectory_digest(
    analysis: TrajectoryAnalysis,
    *,
    max_events: int = 40,
) -> list[str]:
    """Render a compact step-level digest for judge consumption."""
    rows = []
    for observation in analysis.read_observations[:max_events]:
        raw_suffix = (
            f" | command={observation.raw_command}"
            if observation.raw_command is not None
            else ""
        )
        rows.append(
            f"step={observation.step_id} tool={observation.tool_name} "
            f"access={observation.access_type} path={observation.path}{raw_suffix}"
        )
    return rows

