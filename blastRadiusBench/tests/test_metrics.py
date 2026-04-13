from pathlib import Path

from blast_radius_bench.atif import analyze_trajectory
from blast_radius_bench.judge import build_judge_prompt
from blast_radius_bench.metrics import score_analysis
from blast_radius_bench.models import load_task_spec


FIXTURES = Path(__file__).parent / "fixtures"


def test_score_analysis_reports_blast_radius_metrics() -> None:
    task_spec = load_task_spec(FIXTURES / "sample_task.yaml")
    analysis = analyze_trajectory(
        FIXTURES / "sample_trajectory.json",
        repo_root_aliases=task_spec.repo_root_aliases,
    )

    report = score_analysis(task_spec, analysis)

    assert report.required_hits == ["src/app.py"]
    assert report.allowed_hits == ["src/helpers.py"]
    assert report.forbidden_hits == ["docs/architecture.md"]
    assert report.out_of_scope_reads == []
    assert report.required_file_recall == 1.0
    assert report.target_file_recall == 1.0
    assert report.justified_file_precision == 0.6667
    assert report.forbidden_read_rate == 0.3333
    assert report.context_bloat_ratio == 0.0


def test_build_judge_prompt_embeds_digest() -> None:
    task_spec = load_task_spec(FIXTURES / "sample_task.yaml")
    analysis = analyze_trajectory(
        FIXTURES / "sample_trajectory.json",
        repo_root_aliases=task_spec.repo_root_aliases,
    )

    prompt = build_judge_prompt(task_spec, analysis)

    assert "focused" in prompt.user_prompt
    assert "src/app.py" in prompt.user_prompt
    assert "docs/architecture.md" in prompt.user_prompt
