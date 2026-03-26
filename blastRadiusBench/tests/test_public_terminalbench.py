import json

from blast_radius_bench.public_terminalbench import (
    classify_public_tool,
    summarize_public_steps,
)


def test_classify_public_tool_groups_common_tool_names() -> None:
    assert classify_public_tool("bash_command") == "shell"
    assert classify_public_tool("Read") == "read"
    assert classify_public_tool("str_replace_editor") == "edit"
    assert classify_public_tool("TodoWrite") == "plan"
    assert classify_public_tool("mark_task_complete") == "other"


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
    assert summary["tool_counter"]["Read"] == 1
    assert summary["tool_counter"]["bash_command"] == 1
