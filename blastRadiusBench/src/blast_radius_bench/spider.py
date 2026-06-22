"""Heuristic spider plots for harness/model behavior fingerprints."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SpiderSourceType = Literal["auto", "public-terminalbench", "wolfbench"]

TRACE_AXIS_COLUMNS = [
    "success",
    "task_lift",
    "tool_intensity",
    "exploration_diversity",
    "search_read_intensity",
    "edit_eagerness",
    "verification_intensity",
    "recovery_churn",
]

WOLFBENCH_AXIS_COLUMNS = [
    "average",
    "solid",
    "ceiling",
    "best",
    "reliability",
    "variance_gap",
    "error_pressure",
    "speed",
]

AXIS_LABELS = {
    "success": "Success",
    "task_lift": "Task Lift",
    "tool_intensity": "Tool Intensity",
    "exploration_diversity": "Exploration",
    "search_read_intensity": "Search/Read",
    "edit_eagerness": "Early Edit",
    "verification_intensity": "Verification",
    "recovery_churn": "Recovery Churn",
    "average": "Average",
    "solid": "Solid",
    "ceiling": "Ceiling",
    "best": "Best",
    "reliability": "Reliability",
    "variance_gap": "Variance Gap",
    "error_pressure": "Error Pressure",
    "speed": "Speed",
}


def build_spider_report(
    source: str | Path,
    output_dir: str | Path,
    *,
    source_type: SpiderSourceType = "auto",
    min_runs: int = 10,
    min_trace_coverage: float = 0.25,
    top_groups: int = 12,
) -> dict[str, Any]:
    """Build spider-plot CSVs, PNGs, and HTML from an existing report directory."""
    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_type = detect_spider_source_type(source_path, source_type)
    if resolved_type == "public-terminalbench":
        run_df = _read_public_terminalbench_runs(source_path)
        summary_df, axis_columns = build_trace_spider_frame(
            run_df,
            min_runs=min_runs,
            min_trace_coverage=min_trace_coverage,
        )
        mode_label = "Trace Behavior"
        interpretation = (
            "Trace mode is a behavior fingerprint. High values mean more of that "
            "property, not always better. For example, high tool intensity or "
            "recovery churn can be useful evidence of harness/model style."
        )
    elif resolved_type == "wolfbench":
        config_df = _read_wolfbench_config_summary(source_path)
        summary_df, axis_columns = build_wolfbench_spider_frame(config_df, min_runs=min_runs)
        mode_label = "WolfBench Reliability"
        interpretation = (
            "WolfBench mode uses run-level reliability signals. High variance gap "
            "and error pressure are skews, while high average, solid, ceiling, and "
            "reliability are favorable outcome signals."
        )
    else:
        raise ValueError(f"Unsupported spider source type: {resolved_type}")

    summary_df.to_csv(output_path / "spider_summary.csv", index=False)
    long_df = _to_long_axes(summary_df, axis_columns)
    long_df.to_csv(output_path / "spider_axes.csv", index=False)

    plot_paths = _write_spider_plots(
        summary_df=summary_df,
        axis_columns=axis_columns,
        output_dir=output_path,
        top_groups=top_groups,
    )
    html_path = _write_html_report(
        output_dir=output_path,
        source=str(source),
        source_type=resolved_type,
        mode_label=mode_label,
        interpretation=interpretation,
        summary_df=summary_df,
        axis_columns=axis_columns,
        plot_paths=plot_paths,
        min_runs=min_runs,
        min_trace_coverage=min_trace_coverage if resolved_type == "public-terminalbench" else None,
        top_groups=top_groups,
    )

    return {
        "source": str(source),
        "source_type": resolved_type,
        "groups": int(len(summary_df)),
        "axes": axis_columns,
        "output_dir": str(output_path),
        "html_report": str(html_path),
        "plots": [str(path) for path in plot_paths],
    }


def detect_spider_source_type(
    source_path: Path,
    source_type: SpiderSourceType = "auto",
) -> Literal["public-terminalbench", "wolfbench"]:
    """Detect whether a report directory contains trace or WolfBench summaries."""
    if source_type != "auto":
        return source_type

    if source_path.is_file():
        frame = pd.read_csv(source_path, nrows=1)
        columns = set(frame.columns)
        if {"config_name", "average_score", "solid_score"}.issubset(columns):
            return "wolfbench"
        if {"agent", "model", "has_steps", "tool_call_count"}.issubset(columns):
            return "public-terminalbench"

    if (source_path / "config_summary.csv").exists():
        return "wolfbench"
    if (source_path / "run_metrics.csv").exists():
        return "public-terminalbench"

    raise FileNotFoundError(
        "Could not detect spider source. Pass a report directory containing "
        "run_metrics.csv or config_summary.csv."
    )


def build_trace_spider_frame(
    run_df: pd.DataFrame,
    *,
    min_runs: int,
    min_trace_coverage: float = 0.25,
) -> tuple[pd.DataFrame, list[str]]:
    """Build normalized spider axes for public trace run metrics."""
    if run_df.empty:
        return pd.DataFrame(), TRACE_AXIS_COLUMNS

    frame = run_df.copy()
    frame["success_float"] = frame["success"].astype(float)
    task_success = frame.groupby("task_name")["success_float"].transform("mean")
    frame["task_adjusted_success_lift"] = frame["success_float"] - task_success

    agg_spec = {
        "runs": ("agent", "size"),
        "tasks": ("task_name", "nunique"),
        "success_rate": ("success_float", "mean"),
        "task_adjusted_success_lift": ("task_adjusted_success_lift", "mean"),
        "trace_coverage": ("has_steps", "mean"),
        "mean_tool_call_count": ("tool_call_count", "mean"),
        "mean_duration_seconds": ("duration_seconds", "mean"),
        "mean_cost_cents": ("cost_cents", "mean"),
        "mean_first_edit_step": ("first_edit_step", "mean"),
        "mean_command_intent_entropy": ("command_intent_entropy", "mean"),
        "mean_tool_category_entropy": ("tool_category_entropy", "mean"),
        "mean_failure_rate": ("failure_rate", "mean"),
        "mean_repeated_tool_command_rate": ("repeated_tool_command_rate", "mean"),
        "mean_post_failure_tool_calls": ("post_failure_tool_calls", "mean"),
        "mean_search_intent_calls": ("search_intent_calls", "mean"),
        "mean_read_intent_calls": ("read_intent_calls", "mean"),
        "mean_test_intent_calls": ("test_intent_calls", "mean"),
        "mean_build_intent_calls": ("build_intent_calls", "mean"),
    }
    summary = frame.groupby(["agent", "model"], dropna=False).agg(**agg_spec).reset_index()
    summary = summary[
        (summary["runs"] >= min_runs)
        & (summary["trace_coverage"] >= min_trace_coverage)
    ].copy()
    if summary.empty:
        return summary, TRACE_AXIS_COLUMNS

    denominator = summary["mean_tool_call_count"].replace(0, np.nan)
    summary["search_read_share"] = (
        summary["mean_search_intent_calls"].fillna(0)
        + summary["mean_read_intent_calls"].fillna(0)
    ) / denominator
    summary["verification_share"] = (
        summary["mean_test_intent_calls"].fillna(0)
        + summary["mean_build_intent_calls"].fillna(0)
    ) / denominator
    summary["exploration_raw"] = (
        summary["mean_command_intent_entropy"].fillna(0)
        + summary["mean_tool_category_entropy"].fillna(0)
    ) / 2
    summary["recovery_churn_raw"] = (
        summary["mean_failure_rate"].fillna(0)
        + summary["mean_repeated_tool_command_rate"].fillna(0)
        + _scale(summary["mean_post_failure_tool_calls"].fillna(0))
    ) / 3

    summary["success"] = _scale(summary["success_rate"])
    summary["task_lift"] = _scale(summary["task_adjusted_success_lift"])
    summary["tool_intensity"] = _scale(summary["mean_tool_call_count"])
    summary["exploration_diversity"] = _scale(summary["exploration_raw"])
    summary["search_read_intensity"] = _scale(summary["search_read_share"])
    summary["edit_eagerness"] = _scale(summary["mean_first_edit_step"], invert=True)
    summary["verification_intensity"] = _scale(summary["verification_share"])
    summary["recovery_churn"] = _scale(summary["recovery_churn_raw"])
    summary["group_label"] = summary["agent"].astype(str) + " / " + summary["model"].astype(str)
    summary["spider_area"] = summary[TRACE_AXIS_COLUMNS].mean(axis=1)

    sort_columns = ["success_rate", "runs"]
    summary = summary.sort_values(sort_columns, ascending=[False, False])
    return _round_numeric(summary), TRACE_AXIS_COLUMNS


def build_wolfbench_spider_frame(
    config_summary: pd.DataFrame,
    *,
    min_runs: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build normalized spider axes for WolfBench run-level reliability summaries."""
    if config_summary.empty:
        return pd.DataFrame(), WOLFBENCH_AXIS_COLUMNS

    summary = config_summary[config_summary["runs"] >= min_runs].copy()
    if summary.empty:
        return summary, WOLFBENCH_AXIS_COLUMNS

    summary["average"] = _scale(summary["average_score"])
    summary["solid"] = _scale(summary["solid_score"])
    summary["ceiling"] = _scale(summary["ceiling_score"])
    summary["best"] = _scale(summary["best_of_score"])
    summary["reliability"] = _scale(summary["ceiling_minus_solid"], invert=True)
    summary["variance_gap"] = _scale(summary["ceiling_minus_solid"])
    summary["error_pressure"] = _scale(summary["mean_errors"].fillna(0))
    summary["speed"] = _scale(summary["mean_duration_minutes"], invert=True)
    summary["group_label"] = summary["config_name"].astype(str)
    summary["spider_area"] = summary[WOLFBENCH_AXIS_COLUMNS].mean(axis=1)

    summary = summary.sort_values(
        ["average_score", "solid_score", "runs"],
        ascending=[False, False, False],
    )
    return _round_numeric(summary), WOLFBENCH_AXIS_COLUMNS


def _read_public_terminalbench_runs(source_path: Path) -> pd.DataFrame:
    path = source_path if source_path.is_file() else source_path / "run_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing public Terminal-Bench run metrics: {path}")
    return pd.read_csv(path)


def _read_wolfbench_config_summary(source_path: Path) -> pd.DataFrame:
    path = source_path if source_path.is_file() else source_path / "config_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing WolfBench config summary: {path}")
    return pd.read_csv(path)


def _to_long_axes(summary_df: pd.DataFrame, axis_columns: list[str]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["group_label", "axis", "label", "value"])
    id_cols = [
        column
        for column in ["group_label", "agent", "model", "config_name", "runs", "tasks"]
        if column in summary_df.columns
    ]
    long_df = summary_df[id_cols + axis_columns].melt(
        id_vars=id_cols,
        value_vars=axis_columns,
        var_name="axis",
        value_name="value",
    )
    long_df["label"] = long_df["axis"].map(AXIS_LABELS)
    return long_df


def _write_spider_plots(
    *,
    summary_df: pd.DataFrame,
    axis_columns: list[str],
    output_dir: Path,
    top_groups: int,
) -> list[Path]:
    if summary_df.empty:
        return []
    plot_paths = []
    top = summary_df.head(top_groups).copy()
    overlay_count = min(6, len(top))
    if overlay_count:
        plot_paths.append(
            _plot_spider_overlay(
                top.head(overlay_count),
                axis_columns,
                output_dir / "spider_overlay.png",
            )
        )
    plot_paths.append(
        _plot_spider_grid(
            top,
            axis_columns,
            output_dir / "spider_grid.png",
        )
    )
    plot_paths.append(
        _plot_axis_heatmap(
            top,
            axis_columns,
            output_dir / "spider_axis_heatmap.png",
        )
    )
    return plot_paths


def _plot_spider_overlay(
    frame: pd.DataFrame,
    axis_columns: list[str],
    output_path: Path,
) -> Path:
    angles = _radar_angles(len(axis_columns))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for _, row in frame.iterrows():
        values = _closed_values(row, axis_columns)
        ax.plot(angles, values, linewidth=2, label=str(row["group_label"]))
        ax.fill(angles, values, alpha=0.08)
    _style_radar_axis(ax, axis_columns, "Top Harness x Model Fingerprints")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_spider_grid(
    frame: pd.DataFrame,
    axis_columns: list[str],
    output_path: Path,
) -> Path:
    count = len(frame)
    cols = min(3, max(1, count))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.2, rows * 4.0),
        subplot_kw={"projection": "polar"},
    )
    flat_axes = np.atleast_1d(axes).ravel()
    angles = _radar_angles(len(axis_columns))
    for ax, (_, row) in zip(flat_axes, frame.iterrows(), strict=False):
        values = _closed_values(row, axis_columns)
        ax.plot(angles, values, linewidth=2, color="#1d7f64")
        ax.fill(angles, values, alpha=0.18, color="#1d7f64")
        _style_radar_axis(ax, axis_columns, str(row["group_label"]), label_size=7)
    for ax in flat_axes[count:]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_axis_heatmap(
    frame: pd.DataFrame,
    axis_columns: list[str],
    output_path: Path,
) -> Path:
    labels = frame["group_label"].astype(str).tolist()
    values = frame[axis_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, max(4.5, len(frame) * 0.34)))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Spider Axis Heatmap")
    ax.set_xticks(range(len(axis_columns)))
    ax.set_xticklabels([AXIS_LABELS[column] for column in axis_columns], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="Normalized axis value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _radar_angles(count: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def _closed_values(row: pd.Series, axis_columns: list[str]) -> list[float]:
    values = [float(row[column]) if pd.notna(row[column]) else 0.0 for column in axis_columns]
    return [*values, values[0]]


def _style_radar_axis(
    ax: Any,
    axis_columns: list[str],
    title: str,
    *,
    label_size: int = 8,
) -> None:
    angles = _radar_angles(len(axis_columns))[:-1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels([AXIS_LABELS[column] for column in axis_columns], fontsize=label_size)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25", ".50", ".75", "1.0"], fontsize=7)
    ax.set_title(title, y=1.08, fontsize=10)
    ax.grid(alpha=0.35)


def _write_html_report(
    *,
    output_dir: Path,
    source: str,
    source_type: str,
    mode_label: str,
    interpretation: str,
    summary_df: pd.DataFrame,
    axis_columns: list[str],
    plot_paths: list[Path],
    min_runs: int,
    min_trace_coverage: float | None,
    top_groups: int,
) -> Path:
    output_path = output_dir / "index.html"
    plot_lookup = {path.stem: path.name for path in plot_paths}
    plot_html = _plot_grid_html(
        plot_lookup,
        [
            ("spider_overlay", "Top Overlay"),
            ("spider_grid", "Small Multiples"),
            ("spider_axis_heatmap", "Axis Heatmap"),
        ],
    )
    table_columns = [
        column
        for column in [
            "group_label",
            "runs",
            "tasks",
            "success_rate",
            "task_adjusted_success_lift",
            "spider_area",
            *axis_columns,
        ]
        if column in summary_df.columns
    ]
    table_html = _dashboard_table(summary_df, table_columns, limit=max(30, top_groups))
    axis_pills = "".join(
        f"<span class='pill'>{html.escape(AXIS_LABELS[column])}</span>"
        for column in axis_columns
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spider Fingerprints</title>
  <style>
    :root {{
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #62625f;
      --line: #dedbd2;
      --green: #1d7f64;
      --blue: #2e6f9e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); line-height: 1.45; }}
    .shell {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
    header {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 0.45fr); gap: 22px; align-items: end; padding: 28px 0 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: clamp(34px, 5vw, 68px); line-height: 0.95; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    code {{ background: #ece9df; padding: 2px 6px; border-radius: 4px; }}
    .lede {{ max-width: 880px; margin-top: 14px; font-size: 18px; color: #343431; }}
    .meta {{ display: grid; gap: 8px; align-content: end; font-size: 14px; min-width: 0; }}
    .meta div {{ overflow-wrap: anywhere; }}
    .band {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .pillrow {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: 12px; background: #fbfaf6; color: #353530; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .plot-card {{ border: 1px solid var(--line); background: #fff; border-radius: 8px; padding: 12px; }}
    .plot-card img {{ width: 100%; display: block; border-radius: 6px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 860px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #ece9df; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f4f1e8; z-index: 1; color: #3a3935; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
    tr:hover td {{ background: #fbfaf6; }}
    @media (max-width: 980px) {{ .shell {{ padding: 16px; }} header, .plot-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>Spider Fingerprints</h1>
        <p class="lede">{html.escape(mode_label)} spider plots for harness x model cells. {html.escape(interpretation)}</p>
      </div>
      <div class="meta">
        <div>Source <code>{html.escape(source)}</code></div>
        <div>Detected type <code>{html.escape(source_type)}</code></div>
        <div>Minimum runs <code>{min_runs}</code></div>
        <div>Min trace coverage <code>{min_trace_coverage if min_trace_coverage is not None else "n/a"}</code></div>
        <div>Rendered groups <code>{top_groups}</code></div>
      </div>
    </header>
    <section class="band">
      <h2>Axes</h2>
      <p>All axes are normalized to 0..1 within this corpus, so compare shapes inside one report rather than across unrelated datasets.</p>
      <div class="pillrow">{axis_pills}</div>
    </section>
    <section class="band">{plot_html}</section>
    <section class="band"><h2>Spider Summary</h2>{table_html}</section>
  </div>
</body>
</html>
"""
    output_path.write_text(html_doc)
    return output_path


def _plot_grid_html(plot_lookup: dict[str, str], plots: list[tuple[str, str]]) -> str:
    cards = []
    for stem, title in plots:
        filename = plot_lookup.get(stem)
        if not filename:
            continue
        cards.append(
            "<article class='plot-card'>"
            f"<h3>{html.escape(title)}</h3>"
            f"<img src='{html.escape(filename)}' alt='{html.escape(title)}' />"
            "</article>"
        )
    if not cards:
        return "<p>No plots available.</p>"
    return f"<div class='plot-grid'>{''.join(cards)}</div>"


def _dashboard_table(frame: pd.DataFrame, columns: list[str], *, limit: int) -> str:
    if frame.empty:
        return "<p>No rows available.</p>"
    selected = [column for column in columns if column in frame.columns]
    rendered = frame[selected].head(limit).copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(_format_cell)
    return (
        "<div class='table-wrap'>"
        + rendered.to_html(index=False, escape=False, border=0)
        .replace('class="dataframe"', 'class="dataframe dashboard-table"')
        + "</div>"
    )


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if -1 <= value <= 1:
            return f"{value:.3f}"
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _scale(series: pd.Series, *, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        scaled = pd.Series(0.5, index=series.index)
    else:
        filled = values.fillna(values.median())
        minimum = filled.min()
        maximum = filled.max()
        if maximum == minimum:
            scaled = pd.Series(0.5, index=series.index)
        else:
            scaled = (filled - minimum) / (maximum - minimum)
    if invert:
        scaled = 1 - scaled
    return scaled.clip(0, 1)


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_cols = result.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_cols:
        if result[column].dtype == bool:
            continue
        result[column] = result[column].round(4)
    return result
