import json

import pandas as pd

from blast_radius_bench.public_terminalbench import (
    _add_task_adjusted_success,
    _build_agent_contrast_summary,
    _build_same_model_agent_contrast_summary,
    _build_task_agent_contrast_summary,
    classify_public_command_intent,
    classify_public_tool,
    summarize_public_steps,
)


def test_classify_public_tool_groups_common_tool_names() -> None:
    assert classify_public_tool("bash_command") == "shell"
    assert classify_public_tool("execute") == "shell"
    assert classify_public_tool("run_shell_command") == "shell"
    assert classify_public_tool("Read") == "read"
    assert classify_public_tool("str_replace_editor") == "edit"
    assert classify_public_tool("TodoWrite") == "plan"
    assert classify_public_tool("mark_task_complete") == "other"


def test_classify_public_command_intent_uses_shell_command_shape() -> None:
    assert classify_public_command_intent("execute", "bash -lc 'rg needle src'") == "search"
    assert classify_public_command_intent("execute", "cd /app && rg needle src") == "search"
    assert classify_public_command_intent("execute", "pytest tests") == "test"
    assert classify_public_command_intent("execute", "pip install pandas") == "dependency"
    assert classify_public_command_intent("bash_command", "apt-get update && apt-get install -y r-base") == "dependency"
    assert classify_public_command_intent("execute", "cat src/app.py") == "read"
    assert classify_public_command_intent("execute", "") == "other"
    assert classify_public_command_intent("mark_task_complete", "") == "finish"
    assert classify_public_command_intent("bash_command", "C-c") == "control"
    assert classify_public_command_intent("Read", "src/app.py") == "read"
    assert classify_public_command_intent("str_replace_editor", "src/app.py") == "edit"


def test_summarize_public_steps_counts_tools_and_categories() -> None:
    steps_raw = json.dumps(
        [
            {"src": "user", "msg": "warmup", "tools": None, "obs": None},
            {
                "src": "agent",
                "msg": "Read and edit",
                "tools": [
                    {"fn": "Read", "cmd": "/app/src/app.py"},
                    {"fn": "bash_command", "cmd": "ls -la src\n"},
                    {"fn": "str_replace_editor", "cmd": "/app/src/app.py"},
                ],
                "obs": None,
            },
            {
                "src": "agent",
                "msg": "Plan",
                "tools": [{"fn": "TodoWrite", "cmd": ""}],
                "obs": None,
            },
        ]
    )

    summary = summarize_public_steps(steps_raw)

    assert summary["step_count"] == 3
    assert summary["agent_step_count"] == 2
    assert summary["tool_call_count"] == 4
    assert summary["shell_tool_calls"] == 1
    assert summary["read_tool_calls"] == 1
    assert summary["edit_tool_calls"] == 1
    assert summary["plan_tool_calls"] == 1
    assert summary["other_tool_calls"] == 0
    assert summary["first_tool_step"] == 1
    assert summary["first_read_step"] == 1
    assert summary["first_edit_step"] == 1
    assert summary["first_plan_step"] == 2
    assert summary["pre_edit_tool_calls"] == 2
    assert summary["category_switch_rate"] == 1.0
    assert summary["longest_same_category_run"] == 1
    assert summary["repeated_tool_command_rate"] == 0.0
    assert summary["intent_counter"]["read"] == 1
    assert summary["intent_counter"]["inspect"] == 1
    assert summary["intent_counter"]["edit"] == 1
    assert summary["intent_counter"]["plan"] == 1
    assert summary["tool_counter"]["Read"] == 1
    assert summary["tool_counter"]["bash_command"] == 1


def test_summarize_public_steps_tracks_repeated_command_dynamics() -> None:
    steps_raw = json.dumps(
        [
            {
                "src": "agent",
                "tools": [
                    {"fn": "execute", "cmd": "pytest"},
                    {"fn": "execute", "cmd": "pytest"},
                    {"fn": "execute", "cmd": "pytest"},
                    {"fn": "str_replace_editor", "cmd": "src/app.py"},
                ],
            }
        ]
    )

    summary = summarize_public_steps(steps_raw)

    assert summary["shell_tool_calls"] == 3
    assert summary["pre_edit_tool_calls"] == 3
    assert summary["category_switch_rate"] == 1 / 3
    assert summary["longest_same_category_run"] == 3
    assert summary["repeated_tool_command_rate"] == 0.5


def test_summarize_public_steps_tracks_failure_recovery_shape() -> None:
    steps_raw = json.dumps(
        [
            {
                "src": "agent",
                "tools": [{"fn": "execute", "cmd": "pytest"}],
                "obs": "Command failed with exit code 1",
            },
            {
                "src": "agent",
                "tools": [{"fn": "execute", "cmd": "cat src/app.py"}],
                "obs": "ok",
            },
        ]
    )

    summary = summarize_public_steps(steps_raw)

    assert summary["failed_tool_call_count"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["first_failure_step"] == 0
    assert summary["post_failure_tool_calls"] == 1
    assert summary["post_failure_intent_entropy"] == 0.0


def test_build_agent_contrast_summary_finds_similar_outcome_different_process() -> None:
    agent_summary = pd.DataFrame(
        [
            {
                "agent": "narrow",
                "runs": 500,
                "trace_coverage": 0.9,
                "success_rate": 0.5,
                "mean_tool_call_count": 5,
                "mean_category_switch_rate": 0.1,
                "mean_repeated_tool_command_rate": 0.1,
                "mean_tool_category_entropy": 0.2,
            },
            {
                "agent": "wide",
                "runs": 500,
                "trace_coverage": 0.9,
                "success_rate": 0.52,
                "mean_tool_call_count": 40,
                "mean_category_switch_rate": 0.6,
                "mean_repeated_tool_command_rate": 0.4,
                "mean_tool_category_entropy": 1.2,
            },
        ]
    )

    contrast = _build_agent_contrast_summary(agent_summary)

    assert contrast.loc[0, "agent_a"] == "narrow"
    assert contrast.loc[0, "agent_b"] == "wide"
    assert contrast.loc[0, "success_gap"] == 0.02
    assert contrast.loc[0, "tool_call_gap"] == 35


def test_add_task_adjusted_success_computes_weighted_lift() -> None:
    agent_summary = pd.DataFrame([{"agent": "agent-a", "runs": 4}])
    task_agent_summary = pd.DataFrame(
        [
            {"task_name": "easy", "agent": "agent-a", "runs": 3, "success_rate": 1.0},
            {"task_name": "hard", "agent": "agent-a", "runs": 1, "success_rate": 0.0},
        ]
    )
    task_summary = pd.DataFrame(
        [
            {"task_name": "easy", "success_rate": 0.75},
            {"task_name": "hard", "success_rate": 0.25},
        ]
    )

    adjusted = _add_task_adjusted_success(
        agent_summary,
        task_agent_summary=task_agent_summary,
        task_summary=task_summary,
    )

    assert adjusted.loc[0, "task_adjusted_success_lift"] == 0.125
    assert adjusted.loc[0, "task_agent_cells"] == 2


def test_build_task_agent_contrast_summary_compares_within_task() -> None:
    task_agent_summary = pd.DataFrame(
        [
            {
                "task_name": "task-a",
                "agent": "agent-a",
                "runs": 5,
                "trace_coverage": 1.0,
                "success_rate": 0.6,
                "mean_tool_call_count": 5,
                "mean_category_switch_rate": 0.1,
                "mean_repeated_tool_command_rate": 0.1,
                "mean_command_intent_entropy": 0.5,
            },
            {
                "task_name": "task-a",
                "agent": "agent-b",
                "runs": 5,
                "trace_coverage": 1.0,
                "success_rate": 0.65,
                "mean_tool_call_count": 35,
                "mean_category_switch_rate": 0.4,
                "mean_repeated_tool_command_rate": 0.2,
                "mean_command_intent_entropy": 1.5,
            },
        ]
    )

    contrast = _build_task_agent_contrast_summary(task_agent_summary)

    assert contrast.loc[0, "task_name"] == "task-a"
    assert contrast.loc[0, "success_gap"] == 0.05
    assert contrast.loc[0, "tool_call_gap"] == 30


def test_build_same_model_agent_contrast_summary_compares_harnesses() -> None:
    model_summary = pd.DataFrame(
        [
            {
                "model": "same-model",
                "agent": "agent-a",
                "runs": 100,
                "trace_coverage": 1.0,
                "success_rate": 0.5,
                "mean_tool_call_count": 10,
                "mean_category_switch_rate": 0.1,
                "mean_repeated_tool_command_rate": 0.1,
            },
            {
                "model": "same-model",
                "agent": "agent-b",
                "runs": 100,
                "trace_coverage": 1.0,
                "success_rate": 0.7,
                "mean_tool_call_count": 30,
                "mean_category_switch_rate": 0.2,
                "mean_repeated_tool_command_rate": 0.3,
            },
        ]
    )

    contrast = _build_same_model_agent_contrast_summary(model_summary)

    assert contrast.loc[0, "model"] == "same-model"
    assert contrast.loc[0, "agent_a"] == "agent-a"
    assert contrast.loc[0, "agent_b"] == "agent-b"
    assert contrast.loc[0, "tool_call_gap"] == 20
