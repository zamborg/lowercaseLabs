from pathlib import Path

from blast_radius_bench.atif import analyze_trajectory


FIXTURES = Path(__file__).parent / "fixtures"


def test_analyze_trajectory_extracts_reads_and_searches() -> None:
    analysis = analyze_trajectory(FIXTURES / "sample_trajectory.json")

    assert analysis.session_id == "session-1"
    assert analysis.agent_name == "codex"
    assert analysis.total_tool_calls == 3
    assert analysis.total_prompt_tokens == 18
    assert analysis.total_completion_tokens == 8
    assert analysis.total_cost_usd == 0.0015
    assert analysis.unique_paths("content") == [
        "docs/architecture.md",
        "src/app.py",
        "src/helpers.py",
    ]
    assert analysis.unique_paths("search") == ["src"]


def test_analyze_trajectory_extracts_terminus_bash_keystrokes() -> None:
    analysis = analyze_trajectory(FIXTURES / "terminus_trajectory.json")

    assert analysis.session_id == "session-2"
    assert analysis.agent_name == "terminus-2"
    assert analysis.total_tool_calls == 1
    assert analysis.unique_paths("content") == [
        "src/regex.py",
        "tests/test_regex.py",
    ]
    assert analysis.unique_paths("search") == [
        "assets/books",
        "src",
        "tests",
    ]
