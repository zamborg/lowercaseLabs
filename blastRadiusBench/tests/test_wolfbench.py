import json
from pathlib import Path

from blast_radius_bench.wolfbench import (
    build_wolfbench_config_summary,
    build_wolfbench_report,
    build_wolfbench_task_summary,
    load_wolfbench_run_data,
)


def test_load_wolfbench_run_data_flattens_run_and_task_rewards(tmp_path: Path) -> None:
    runs_root = tmp_path / "wolfbench-runs"
    _write_run(
        runs_root,
        config_name="t2-example",
        run_name="2026-03-01__00-00-00",
        rewards={"1.0": ["task-a__aaa", "task-b__bbb"], "0.0": ["task-c__ccc"]},
        score=2 / 3,
    )

    run_df, task_run_df = load_wolfbench_run_data(runs_root)

    assert len(run_df) == 1
    assert len(task_run_df) == 3
    assert run_df.loc[0, "agent"] == "terminus-2"
    assert run_df.loc[0, "model"] == "openai/example-model"
    assert run_df.loc[0, "solved_tasks"] == 2
    assert sorted(task_run_df["task_name"].tolist()) == ["task-a", "task-b", "task-c"]
    assert task_run_df.set_index("task_name").loc["task-c", "success"].item() is False


def test_build_wolfbench_config_summary_computes_five_metrics(tmp_path: Path) -> None:
    runs_root = tmp_path / "wolfbench-runs"
    _write_run(
        runs_root,
        config_name="t2-example",
        run_name="2026-03-01__00-00-00",
        rewards={"1.0": ["task-a__aaa", "task-b__bbb"], "0.0": ["task-c__ccc"]},
        score=2 / 3,
    )
    _write_run(
        runs_root,
        config_name="t2-example",
        run_name="2026-03-01__01-00-00",
        rewards={"1.0": ["task-a__ddd"], "0.0": ["task-b__eee", "task-c__fff"]},
        score=1 / 3,
    )

    run_df, task_run_df = load_wolfbench_run_data(runs_root)
    summary = build_wolfbench_config_summary(run_df, task_run_df).set_index("config_name")
    row = summary.loc["t2-example"]

    assert row["runs"] == 2
    assert row["tasks"] == 3
    assert row["average_score"] == 0.5
    assert row["best_of_score"] == 0.6667
    assert row["worst_of_score"] == 0.3333
    assert row["ceiling_score"] == 0.6667
    assert row["solid_score"] == 0.3333
    assert row["ceiling_minus_solid"] == 0.3333


def test_build_wolfbench_report_writes_artifacts(tmp_path: Path) -> None:
    runs_root = tmp_path / "wolfbench-runs"
    _write_run(
        runs_root,
        config_name="t2-example",
        run_name="2026-03-01__00-00-00",
        rewards={"1.0": ["task-a__aaa"], "0.0": ["task-b__bbb"]},
        score=0.5,
    )
    _write_run(
        runs_root,
        config_name="t2-example",
        run_name="2026-03-01__01-00-00",
        rewards={"1.0": ["task-a__ccc", "task-b__ddd"]},
        score=1.0,
    )
    output_dir = tmp_path / "report"

    payload = build_wolfbench_report(runs_root, output_dir)

    assert payload["runs"] == 2
    assert payload["tasks"] == 2
    assert (output_dir / "index.html").exists()
    assert (output_dir / "config_summary.csv").exists()
    assert (output_dir / "task_run_metrics.csv").exists()


def test_build_wolfbench_task_summary_marks_hard_tasks(tmp_path: Path) -> None:
    runs_root = tmp_path / "wolfbench-runs"
    _write_run(
        runs_root,
        config_name="t2-example",
        run_name="2026-03-01__00-00-00",
        rewards={"1.0": ["easy__aaa"], "0.0": ["hard__bbb"]},
        score=0.5,
    )
    _write_run(
        runs_root,
        config_name="oc-example",
        run_name="2026-03-01__00-00-00",
        rewards={"1.0": ["easy__ccc"], "0.0": ["hard__ddd"]},
        score=0.5,
    )

    _, task_run_df = load_wolfbench_run_data(runs_root)
    summary = build_wolfbench_task_summary(task_run_df)

    assert summary.iloc[0]["task_name"] == "hard"
    assert summary.iloc[0]["success_rate"] == 0.0
    assert summary.iloc[1]["task_name"] == "easy"
    assert summary.iloc[1]["success_rate"] == 1.0


def _write_run(
    runs_root: Path,
    *,
    config_name: str,
    run_name: str,
    rewards: dict[str, list[str]],
    score: float,
) -> None:
    run_dir = runs_root / config_name / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "name": "terminus-2",
                        "model_name": "openai/example-model",
                    }
                ],
                "model_display": "Example Model",
                "thinking_display": "off",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "id": f"{config_name}-{run_name}",
                "started_at": "2026-03-01T00:00:00",
                "finished_at": "2026-03-01T00:30:00",
                "n_total_trials": sum(len(values) for values in rewards.values()),
                "stats": {
                    "n_trials": sum(len(values) for values in rewards.values()),
                    "n_errors": 0,
                    "evals": {
                        "terminus-2__example-model__terminal-bench": {
                            "n_trials": sum(len(values) for values in rewards.values()),
                            "n_errors": 0,
                            "metrics": [{"mean": score}],
                            "reward_stats": {"reward": rewards},
                            "exception_stats": {},
                        }
                    },
                },
            }
        )
    )
