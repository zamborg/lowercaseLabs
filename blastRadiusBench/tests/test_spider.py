from pathlib import Path

import pandas as pd

from blast_radius_bench.spider import (
    TRACE_AXIS_COLUMNS,
    WOLFBENCH_AXIS_COLUMNS,
    build_spider_report,
    build_trace_spider_frame,
    build_wolfbench_spider_frame,
)


def test_build_trace_spider_frame_creates_harness_model_axes() -> None:
    run_df = _trace_runs()

    summary, axis_columns = build_trace_spider_frame(run_df, min_runs=2)

    assert axis_columns == TRACE_AXIS_COLUMNS
    assert set(summary["group_label"]) == {"agent-a / model-x", "agent-b / model-x"}
    for axis in axis_columns:
        assert summary[axis].between(0, 1).all()
    assert "success_rate" in summary.columns
    assert "task_adjusted_success_lift" in summary.columns


def test_build_wolfbench_spider_frame_preserves_raw_scores() -> None:
    config_summary = pd.DataFrame(
        [
            {
                "config_name": "agent-a-model-x",
                "runs": 3,
                "tasks": 89,
                "average_score": 0.8,
                "solid_score": 0.7,
                "ceiling_score": 0.9,
                "best_of_score": 0.85,
                "ceiling_minus_solid": 0.2,
                "mean_errors": 1,
                "mean_duration_minutes": 20,
            },
            {
                "config_name": "agent-b-model-x",
                "runs": 3,
                "tasks": 89,
                "average_score": 0.5,
                "solid_score": 0.3,
                "ceiling_score": 0.8,
                "best_of_score": 0.7,
                "ceiling_minus_solid": 0.5,
                "mean_errors": 4,
                "mean_duration_minutes": 40,
            },
        ]
    )

    summary, axis_columns = build_wolfbench_spider_frame(config_summary, min_runs=2)

    assert axis_columns == WOLFBENCH_AXIS_COLUMNS
    assert "average" in summary.columns
    assert summary.set_index("config_name").loc["agent-a-model-x", "average_score"] == 0.8
    for axis in axis_columns:
        assert summary[axis].between(0, 1).all()


def test_build_spider_report_detects_public_terminalbench_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _trace_runs().to_csv(source_dir / "run_metrics.csv", index=False)
    output_dir = tmp_path / "spider"

    payload = build_spider_report(source_dir, output_dir, min_runs=2, top_groups=2)

    assert payload["source_type"] == "public-terminalbench"
    assert payload["groups"] == 2
    assert len(payload["axes"]) == 8
    assert (output_dir / "index.html").exists()
    assert (output_dir / "spider_summary.csv").exists()
    assert (output_dir / "spider_axes.csv").exists()
    assert (output_dir / "spider_grid.png").exists()


def _trace_runs() -> pd.DataFrame:
    rows = []
    for agent, model, task, success, tool_calls, first_edit, entropy, failures in [
        ("agent-a", "model-x", "task-1", True, 10, 2, 0.8, 0.1),
        ("agent-a", "model-x", "task-2", True, 12, 3, 0.9, 0.0),
        ("agent-b", "model-x", "task-1", False, 40, 12, 1.6, 0.3),
        ("agent-b", "model-x", "task-2", True, 45, 14, 1.8, 0.2),
    ]:
        rows.append(
            {
                "dataset_id": "fixture",
                "task_name": task,
                "agent": agent,
                "model": model,
                "reward": 1.0 if success else 0.0,
                "success": success,
                "duration_seconds": tool_calls * 2,
                "input_tokens": tool_calls * 100,
                "output_tokens": tool_calls * 10,
                "cache_tokens": 0,
                "cost_cents": tool_calls / 10,
                "trial_name": f"{task}-{agent}",
                "trial_id": f"{task}-{agent}",
                "has_steps": True,
                "step_count": tool_calls + 1,
                "agent_step_count": tool_calls,
                "tool_call_count": tool_calls,
                "shell_tool_calls": tool_calls,
                "read_tool_calls": 1,
                "edit_tool_calls": 1,
                "plan_tool_calls": 0,
                "other_tool_calls": 0,
                "first_tool_step": 0,
                "first_shell_step": 0,
                "first_read_step": 1,
                "first_edit_step": first_edit,
                "first_plan_step": None,
                "pre_edit_tool_calls": first_edit,
                "category_switch_rate": 0.4,
                "longest_same_category_run": 3,
                "repeated_tool_command_rate": failures / 2,
                "tool_category_entropy": entropy / 2,
                "command_intent_entropy": entropy,
                "intent_switch_rate": 0.3,
                "failed_tool_call_count": int(failures * tool_calls),
                "failure_rate": failures,
                "first_failure_step": 4 if failures else None,
                "post_failure_tool_calls": failures * 10,
                "post_failure_intent_entropy": failures,
                "inspect_intent_calls": 2,
                "search_intent_calls": tool_calls / 5,
                "read_intent_calls": tool_calls / 5,
                "edit_intent_calls": 1,
                "test_intent_calls": tool_calls / 10,
                "build_intent_calls": 0,
                "dependency_intent_calls": 0,
                "run_intent_calls": 0,
                "plan_intent_calls": 0,
                "finish_intent_calls": 1,
                "control_intent_calls": 0,
                "other_intent_calls": 0,
            }
        )
    return pd.DataFrame(rows)
