"""Deterministic scoring for blastRadiusBench."""

from __future__ import annotations

from fnmatch import fnmatch

from blast_radius_bench.models import ScoreReport, TaskSpec, TrajectoryAnalysis


def score_analysis(task_spec: TaskSpec, analysis: TrajectoryAnalysis) -> ScoreReport:
    """Score one analyzed trajectory against one task specification."""
    file_reads = analysis.unique_paths("content")
    search_paths = analysis.unique_paths("search")

    required_hits = sorted(
        path for path in task_spec.gold_context.required_files if path in file_reads
    )
    missing_required = sorted(
        path for path in task_spec.gold_context.required_files if path not in file_reads
    )
    allowed_hits = sorted(
        path for path in task_spec.gold_context.allowed_files if path in file_reads
    )
    forbidden_hits = sorted(path for path in file_reads if _is_forbidden(task_spec, path))
    target_hits = sorted(
        path for path in task_spec.gold_context.target_files if path in file_reads
    )
    missing_targets = sorted(
        path for path in task_spec.gold_context.target_files if path not in file_reads
    )

    justified_reads = sorted(path for path in file_reads if _is_justified(task_spec, path))
    out_of_scope_reads = sorted(
        path
        for path in file_reads
        if path not in justified_reads and path not in forbidden_hits
    )

    total_file_reads = len(file_reads)

    return ScoreReport(
        task_id=task_spec.task_id,
        dataset=task_spec.dataset,
        session_id=analysis.session_id,
        agent_name=analysis.agent_name,
        model_name=analysis.model_name,
        unique_file_reads=file_reads,
        unique_search_paths=search_paths,
        required_hits=required_hits,
        missing_required=missing_required,
        allowed_hits=allowed_hits,
        forbidden_hits=forbidden_hits,
        out_of_scope_reads=out_of_scope_reads,
        target_hits=target_hits,
        missing_targets=missing_targets,
        required_file_recall=_safe_ratio(
            len(required_hits),
            len(task_spec.gold_context.required_files),
        ),
        target_file_recall=_safe_ratio(
            len(target_hits),
            len(task_spec.gold_context.target_files),
        ),
        justified_file_precision=_safe_ratio(len(justified_reads), total_file_reads),
        forbidden_read_rate=_safe_ratio(len(forbidden_hits), total_file_reads),
        context_bloat_ratio=_safe_ratio(len(out_of_scope_reads), total_file_reads),
        total_agent_steps=analysis.total_agent_steps,
        total_tool_calls=analysis.total_tool_calls,
        tool_call_histogram=analysis.tool_call_histogram,
        total_prompt_tokens=analysis.total_prompt_tokens,
        total_completion_tokens=analysis.total_completion_tokens,
        total_cached_tokens=analysis.total_cached_tokens,
        total_cost_usd=analysis.total_cost_usd,
    )


def _is_justified(task_spec: TaskSpec, path: str) -> bool:
    gold = task_spec.gold_context
    if path in gold.required_files or path in gold.allowed_files:
        return True
    return any(fnmatch(path, pattern) for pattern in gold.allowed_globs)


def _is_forbidden(task_spec: TaskSpec, path: str) -> bool:
    gold = task_spec.gold_context
    if path in gold.forbidden_files:
        return True
    return any(fnmatch(path, pattern) for pattern in gold.forbidden_globs)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)

