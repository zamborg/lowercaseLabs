"""Review Harbor trial and job outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blast_radius_bench.atif import analyze_trajectory
from blast_radius_bench.judge import build_judge_prompt
from blast_radius_bench.metrics import score_analysis
from blast_radius_bench.models import TaskSpec


def review_trial(
    trial_dir: str | Path,
    task_spec: TaskSpec,
    *,
    include_prompt: bool = False,
) -> dict[str, Any]:
    """Return a consolidated review payload for one Harbor trial directory."""
    trial_path = Path(trial_dir)
    result_path = trial_path / "result.json"
    payload = json.loads(result_path.read_text())

    exception_info = payload.get("exception_info")
    verifier_result = payload.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    reward = None
    if isinstance(rewards, dict):
        reward = rewards.get("reward")

    trajectory_path = trial_path / "agent" / "trajectory.json"
    score_report = None
    judge_prompt = None
    if trajectory_path.exists():
        analysis = analyze_trajectory(
            trajectory_path,
            repo_root_aliases=task_spec.repo_root_aliases,
        )
        score_report = score_analysis(task_spec, analysis).model_dump(mode="json")
        if include_prompt:
            judge_prompt = build_judge_prompt(task_spec, analysis).model_dump(mode="json")

    success = _is_successful(reward, exception_info)

    return {
        "trial_name": payload.get("trial_name", trial_path.name),
        "task_name": payload.get("task_name"),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "success": success,
        "reward": reward,
        "exception_type": (
            exception_info.get("exception_type") if isinstance(exception_info, dict) else None
        ),
        "exception_message": (
            exception_info.get("exception_message") if isinstance(exception_info, dict) else None
        ),
        "trajectory_path": (
            str(trajectory_path) if trajectory_path.exists() else None
        ),
        "score_report": score_report,
        "judge_prompt": judge_prompt,
    }


def review_job(
    job_dir: str | Path,
    task_spec: TaskSpec,
    *,
    include_prompt: bool = False,
) -> dict[str, Any]:
    """Return a consolidated review payload for one Harbor job directory."""
    job_path = Path(job_dir)
    trial_dirs = sorted(
        path
        for path in job_path.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    )

    trials = [
        review_trial(trial_dir, task_spec, include_prompt=include_prompt)
        for trial_dir in trial_dirs
    ]

    return {
        "job_dir": str(job_path),
        "task_id": task_spec.task_id,
        "dataset": task_spec.dataset,
        "trial_count": len(trials),
        "success_count": sum(1 for trial in trials if trial["success"]),
        "failure_count": sum(1 for trial in trials if not trial["success"]),
        "trials": trials,
    }


def _is_successful(reward: Any, exception_info: Any) -> bool:
    if isinstance(exception_info, dict):
        return False
    try:
        return reward is not None and float(reward) > 0.0
    except (TypeError, ValueError):
        return False

