"""ATIF loading and normalization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
import shlex

from harbor.models.trajectories import ToolCall, Trajectory

from blast_radius_bench.models import ReadObservation, TrajectoryAnalysis
from blast_radius_bench.paths import DEFAULT_REPO_ROOT_ALIASES, normalize_repo_path

_SHELL_TOOL_NAMES = {
    "bash",
    "bash_command",
    "exec_command",
    "execute_bash",
    "run_command",
    "shell",
    "terminal",
}

_CONTENT_TOOL_HINTS = ("open", "read", "view", "cat")
_SEARCH_TOOL_HINTS = ("find", "grep", "list", "ls", "rg", "search", "tree")

_CONTENT_COMMANDS = {"awk", "cat", "head", "less", "more", "nl", "sed", "tail"}
_SEARCH_COMMANDS = {"fd", "find", "grep", "ls", "rg", "tree"}
_COMMAND_ARGUMENT_KEYS = ("cmd", "command", "input", "keystrokes", "shell_command")
_REDIRECTION_OPERATORS = {
    "<",
    "<<",
    ">",
    ">>",
    "1>",
    "1>>",
    "2>",
    "2>>",
    "&>",
    "&>>",
}


def load_trajectory(path: str | Path) -> Trajectory:
    """Load an ATIF trajectory JSON file."""
    return Trajectory.model_validate_json(Path(path).read_text())


def analyze_trajectory(
    path: str | Path,
    *,
    repo_root_aliases: Sequence[str] = DEFAULT_REPO_ROOT_ALIASES,
) -> TrajectoryAnalysis:
    """Extract normalized repo-access data from an ATIF trajectory."""
    trajectory_path = Path(path)
    trajectory = load_trajectory(trajectory_path)

    read_observations: list[ReadObservation] = []
    tool_histogram: Counter[str] = Counter()
    total_tool_calls = 0
    total_agent_steps = 0

    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    cost_usd = 0.0

    for step in trajectory.steps:
        if step.source != "agent":
            continue

        total_agent_steps += 1

        if step.metrics is not None:
            prompt_tokens += step.metrics.prompt_tokens or 0
            completion_tokens += step.metrics.completion_tokens or 0
            cached_tokens += step.metrics.cached_tokens or 0
            cost_usd += step.metrics.cost_usd or 0.0

        for tool_call in step.tool_calls or []:
            total_tool_calls += 1
            tool_histogram[tool_call.function_name] += 1
            read_observations.extend(
                _extract_read_observations(
                    step.step_id,
                    tool_call,
                    repo_root_aliases=repo_root_aliases,
                )
            )

    if trajectory.final_metrics is not None:
        prompt_tokens = trajectory.final_metrics.total_prompt_tokens or prompt_tokens
        completion_tokens = (
            trajectory.final_metrics.total_completion_tokens or completion_tokens
        )
        cached_tokens = trajectory.final_metrics.total_cached_tokens or cached_tokens
        cost_usd = trajectory.final_metrics.total_cost_usd or cost_usd

    deduped = _dedupe_observations(read_observations)

    return TrajectoryAnalysis(
        session_id=trajectory.session_id,
        schema_version=trajectory.schema_version,
        agent_name=trajectory.agent.name,
        model_name=trajectory.agent.model_name,
        trajectory_path=str(trajectory_path),
        total_agent_steps=total_agent_steps,
        total_tool_calls=total_tool_calls,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_cached_tokens=cached_tokens,
        total_cost_usd=round(cost_usd, 8),
        tool_call_histogram=dict(sorted(tool_histogram.items())),
        read_observations=deduped,
    )


def _dedupe_observations(observations: Iterable[ReadObservation]) -> list[ReadObservation]:
    seen: set[tuple[int, str, str, str]] = set()
    deduped: list[ReadObservation] = []
    for observation in observations:
        key = (
            observation.step_id,
            observation.tool_name,
            observation.access_type,
            observation.path,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(observation)
    return deduped


def _extract_read_observations(
    step_id: int,
    tool_call: ToolCall,
    *,
    repo_root_aliases: Sequence[str],
) -> list[ReadObservation]:
    tool_name = tool_call.function_name
    lower_name = tool_name.lower()

    if lower_name in _SHELL_TOOL_NAMES:
        command = _extract_command_argument(tool_call.arguments)
        if not command:
            return []
        return _extract_shell_observations(
            step_id,
            tool_name,
            command,
            repo_root_aliases=repo_root_aliases,
        )

    access_type = _infer_structured_access_type(lower_name)
    if access_type is None:
        return []

    paths = []
    for path_value in _iter_structured_path_values(tool_call.arguments):
        normalized = normalize_repo_path(
            path_value,
            repo_root_aliases,
            allow_bare=True,
        )
        if normalized is not None:
            paths.append(normalized)

    return [
        ReadObservation(
            step_id=step_id,
            tool_name=tool_name,
            access_type=access_type,
            path=path_value,
        )
        for path_value in sorted(dict.fromkeys(paths))
    ]


def _extract_command_argument(arguments: dict[str, object]) -> str | None:
    for key in _COMMAND_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _infer_structured_access_type(tool_name: str) -> str | None:
    if any(hint in tool_name for hint in _CONTENT_TOOL_HINTS):
        return "content"
    if any(hint in tool_name for hint in _SEARCH_TOOL_HINTS):
        return "search"
    return None


def _iter_structured_path_values(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            lower_key = key.lower()
            if lower_key.endswith("_path") or lower_key.endswith("_file") or lower_key in {
                "file",
                "filename",
                "filepath",
                "path",
                "paths",
                "target",
                "target_path",
            }:
                yield from _iter_structured_path_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_structured_path_values(item)
    elif isinstance(value, str):
        yield value


def _extract_shell_observations(
    step_id: int,
    tool_name: str,
    command: str,
    *,
    repo_root_aliases: Sequence[str],
) -> list[ReadObservation]:
    observations: list[ReadObservation] = []
    for segment in _split_shell_segments(command):
        try:
            tokens = shlex.split(segment, comments=False, posix=True)
        except ValueError:
            continue
        tokens = _strip_env_assignments(tokens)
        if not tokens:
            continue

        command_name = Path(tokens[0]).name
        args = tokens[1:]
        access_type, raw_paths = _extract_shell_paths(command_name, args)
        if access_type is None:
            continue

        for raw_path in raw_paths:
            normalized = normalize_repo_path(
                raw_path,
                repo_root_aliases,
                allow_bare=True,
            )
            if normalized is None:
                continue
            observations.append(
                ReadObservation(
                    step_id=step_id,
                    tool_name=tool_name,
                    access_type=access_type,
                    path=normalized,
                    raw_command=command,
                )
            )
    return observations


def _split_shell_segments(command: str) -> list[str]:
    for separator in ("&&", "||", ";", "|"):
        command = command.replace(separator, "\n")
    return [segment.strip() for segment in command.splitlines() if segment.strip()]


def _strip_env_assignments(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if "=" not in token or token.startswith(("./", "/", "~/")):
            break
        if token.count("=") != 1:
            break
        left, _ = token.split("=", 1)
        if not left.isidentifier():
            break
        index += 1
    return tokens[index:]


def _extract_shell_paths(command_name: str, args: list[str]) -> tuple[str | None, list[str]]:
    sanitized = _strip_redirections(args)
    positional = [token for token in sanitized if token and not token.startswith("-")]
    if not positional:
        return None, []

    if command_name in _CONTENT_COMMANDS:
        if command_name in {"sed", "awk"}:
            return "content", positional[-1:]
        return "content", positional

    if command_name in _SEARCH_COMMANDS:
        if command_name in {"rg", "grep", "fd"} and len(positional) > 1:
            return "search", positional[1:]
        if command_name == "find":
            return "search", positional[:1]
        return "search", positional

    return None, []


def _strip_redirections(args: list[str]) -> list[str]:
    sanitized: list[str] = []
    skip_next = False

    for token in args:
        if skip_next:
            skip_next = False
            continue

        if token in _REDIRECTION_OPERATORS:
            skip_next = True
            continue

        if token.startswith((">", "<")):
            continue

        sanitized.append(token)

    return sanitized
