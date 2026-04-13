"""CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from blast_radius_bench.atif import analyze_trajectory
from blast_radius_bench.judge import build_judge_prompt
from blast_radius_bench.metrics import score_analysis
from blast_radius_bench.models import load_task_spec
from blast_radius_bench.public_terminalbench import build_public_terminalbench_report
from blast_radius_bench.review import review_job, review_trial

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def score(
    trajectory: Path,
    spec: Path,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the score report as JSON.",
    ),
) -> None:
    """Score one ATIF trajectory against one task spec."""
    task_spec = load_task_spec(spec)
    analysis = analyze_trajectory(
        trajectory,
        repo_root_aliases=task_spec.repo_root_aliases,
    )
    report = score_analysis(task_spec, analysis)
    payload = report.model_dump(mode="json")
    if json_output:
        console.print_json(json.dumps(payload))
        return
    console.print_json(json.dumps(payload))


@app.command()
def prompt(
    trajectory: Path,
    spec: Path,
) -> None:
    """Render the default judge prompt for one trajectory/task pair."""
    task_spec = load_task_spec(spec)
    analysis = analyze_trajectory(
        trajectory,
        repo_root_aliases=task_spec.repo_root_aliases,
    )
    prompt_payload = build_judge_prompt(task_spec, analysis)
    console.print("[bold]System prompt[/bold]")
    console.print(prompt_payload.system_prompt)
    console.print("\n[bold]User prompt[/bold]")
    console.print(prompt_payload.user_prompt)


@app.command("review-trial")
def review_trial_cmd(
    trial_dir: Path,
    spec: Path,
    include_prompt: bool = typer.Option(
        False,
        "--include-prompt",
        help="Include the generated judge prompt in the output.",
    ),
) -> None:
    """Review one Harbor trial directory."""
    task_spec = load_task_spec(spec)
    payload = review_trial(trial_dir, task_spec, include_prompt=include_prompt)
    console.print_json(json.dumps(payload))


@app.command("review-job")
def review_job_cmd(
    job_dir: Path,
    spec: Path,
    include_prompt: bool = typer.Option(
        False,
        "--include-prompt",
        help="Include generated judge prompts for trials with trajectories.",
    ),
) -> None:
    """Review one Harbor job directory."""
    task_spec = load_task_spec(spec)
    payload = review_job(job_dir, task_spec, include_prompt=include_prompt)
    console.print_json(json.dumps(payload))


@app.command("tb-public-report")
def tb_public_report_cmd(
    dataset_id: str = typer.Argument(
        "yoonholee/terminalbench-trajectories",
        help="Public Hugging Face dataset id to analyze.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/terminalbench-public"),
        "--output-dir",
        help="Directory for HTML, CSV, and plot outputs.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split to analyze.",
    ),
    max_rows: int | None = typer.Option(
        None,
        "--max-rows",
        help="Optional row cap for faster iteration.",
    ),
    min_runs: int = typer.Option(
        100,
        "--min-runs",
        help="Minimum runs required for some aggregate plots.",
    ),
    top_agents: int = typer.Option(
        12,
        "--top-agents",
        help="How many agents to surface in top-agent plots.",
    ),
) -> None:
    """Build a static report for a public Terminal-Bench rollout dataset."""
    payload = build_public_terminalbench_report(
        dataset_id,
        output_dir,
        split=split,
        max_rows=max_rows,
        min_runs=min_runs,
        top_agents=top_agents,
    )
    console.print_json(json.dumps(payload))
