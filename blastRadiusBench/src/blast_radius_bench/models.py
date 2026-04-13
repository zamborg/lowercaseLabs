"""Benchmark data models."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from blast_radius_bench.paths import DEFAULT_REPO_ROOT_ALIASES, normalize_spec_path


class GoldContext(BaseModel):
    """File-level gold context annotations for one task."""

    required_files: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    allowed_globs: list[str] = Field(default_factory=list)
    forbidden_files: list[str] = Field(default_factory=list)
    forbidden_globs: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)

    @field_validator(
        "required_files",
        "allowed_files",
        "forbidden_files",
        "target_files",
        mode="before",
    )
    @classmethod
    def normalize_paths(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        else:
            items = list(value)
        normalized = [normalize_spec_path(str(item)) for item in items]
        return sorted(dict.fromkeys(normalized))

    @field_validator("allowed_globs", "forbidden_globs", mode="before")
    @classmethod
    def normalize_globs(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        items = [str(item).strip() for item in value if str(item).strip()]
        return sorted(dict.fromkeys(items))


class TaskSpec(BaseModel):
    """Task-level benchmark annotation."""

    task_id: str
    dataset: str | None = None
    description: str | None = None
    repo_root_aliases: list[str] = Field(
        default_factory=lambda: list(DEFAULT_REPO_ROOT_ALIASES)
    )
    gold_context: GoldContext

    @field_validator("repo_root_aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: object) -> list[str]:
        if value is None:
            return list(DEFAULT_REPO_ROOT_ALIASES)
        if isinstance(value, str):
            value = [value]
        aliases = [str(item).rstrip("/") for item in value if str(item).strip()]
        return list(dict.fromkeys(aliases))


class ReadObservation(BaseModel):
    """Normalized repo-access event extracted from an ATIF trajectory."""

    step_id: int
    tool_name: str
    access_type: Literal["content", "search"]
    path: str
    raw_command: str | None = None


class TrajectoryAnalysis(BaseModel):
    """Normalized information extracted from a trajectory."""

    session_id: str
    schema_version: str
    agent_name: str
    model_name: str | None = None
    trajectory_path: str | None = None
    total_agent_steps: int = 0
    total_tool_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    tool_call_histogram: dict[str, int] = Field(default_factory=dict)
    read_observations: list[ReadObservation] = Field(default_factory=list)

    def unique_paths(self, access_type: Literal["content", "search"] | None = None) -> list[str]:
        values: Iterable[str]
        if access_type is None:
            values = (observation.path for observation in self.read_observations)
        else:
            values = (
                observation.path
                for observation in self.read_observations
                if observation.access_type == access_type
            )
        return sorted(dict.fromkeys(values))


class ScoreReport(BaseModel):
    """Deterministic benchmark report for one trajectory/task pair."""

    task_id: str
    dataset: str | None = None
    session_id: str
    agent_name: str
    model_name: str | None = None
    unique_file_reads: list[str]
    unique_search_paths: list[str]
    required_hits: list[str]
    missing_required: list[str]
    allowed_hits: list[str]
    forbidden_hits: list[str]
    out_of_scope_reads: list[str]
    target_hits: list[str]
    missing_targets: list[str]
    required_file_recall: float | None
    target_file_recall: float | None
    justified_file_precision: float | None
    forbidden_read_rate: float | None
    context_bloat_ratio: float | None
    total_agent_steps: int
    total_tool_calls: int
    tool_call_histogram: dict[str, int]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cost_usd: float


class JudgePrompt(BaseModel):
    """Prompt pair for an LLM judge."""

    system_prompt: str
    user_prompt: str


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load a YAML task specification."""
    content = Path(path).read_text()
    payload = yaml.safe_load(content)
    return TaskSpec.model_validate(payload)
