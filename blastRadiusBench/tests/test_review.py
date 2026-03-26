from __future__ import annotations

import json
from pathlib import Path

from blast_radius_bench.models import load_task_spec
from blast_radius_bench.review import review_job, review_trial


FIXTURES = Path(__file__).parent / "fixtures"


def test_review_trial_includes_reward_and_score_report(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)

    (agent_dir / "trajectory.json").write_text(
        (FIXTURES / "sample_trajectory.json").read_text()
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "sample-trial",
                "task_name": "sample.single-file-fix",
                "started_at": "2026-03-10T00:00:00Z",
                "finished_at": "2026-03-10T00:00:10Z",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": None,
            }
        )
    )

    task_spec = load_task_spec(FIXTURES / "sample_task.yaml")
    payload = review_trial(trial_dir, task_spec)

    assert payload["success"] is True
    assert payload["reward"] == 1.0
    assert payload["score_report"]["required_file_recall"] == 1.0
    assert payload["score_report"]["forbidden_read_rate"] == 0.3333


def test_review_job_aggregates_trials(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial-a"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)

    (agent_dir / "trajectory.json").write_text(
        (FIXTURES / "sample_trajectory.json").read_text()
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "trial-a",
                "task_name": "sample.single-file-fix",
                "started_at": "2026-03-10T00:00:00Z",
                "finished_at": "2026-03-10T00:00:10Z",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": None,
            }
        )
    )

    task_spec = load_task_spec(FIXTURES / "sample_task.yaml")
    payload = review_job(job_dir, task_spec)

    assert payload["trial_count"] == 1
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 0
