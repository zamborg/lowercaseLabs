from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


transcription_sessions_total = Counter(
    "blackhole_transcription_sessions_total",
    "Total transcription sessions processed",
    ["status"],
)
transcription_latency_seconds = Histogram(
    "blackhole_transcription_latency_seconds",
    "Time spent processing transcription sessions",
)
route_distribution_total = Counter(
    "blackhole_route_distribution_total",
    "Distribution of routing outcomes",
    ["route"],
)
low_confidence_total = Counter(
    "blackhole_low_confidence_total",
    "Sessions routed below the auto-commit threshold",
)
search_latency_seconds = Histogram(
    "blackhole_search_latency_seconds",
    "Search latency in seconds",
)
zero_result_search_total = Counter(
    "blackhole_zero_result_search_total",
    "Searches that returned zero results",
)
note_creation_total = Counter(
    "blackhole_note_creation_total",
    "Notes created",
)
todo_creation_total = Counter(
    "blackhole_todo_creation_total",
    "Todos created",
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

