import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = (
    PROJECT_ROOT / "outputs" / "bert-movie-genres" / "training_history.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PRIMARY_METRICS = ("macro_f1", "micro_f1")
METRIC_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two completed movie-genre training runs."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to training_history.jsonl",
    )
    parser.add_argument(
        "--current",
        help="Current run ID. Defaults to the newest completed run.",
    )
    parser.add_argument(
        "--previous",
        help="Previous run ID. Defaults to the completed run before current.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which comparison reports are created.",
    )
    return parser.parse_args()


def load_completed_runs(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Training history not found at {history_path}. Run train.py first."
        )

    runs_by_id: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            run = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {history_path} at line {line_number}: {error}"
            ) from error
        if run.get("status") == "completed" and run.get("run_id"):
            runs_by_id[run["run_id"]] = run

    return sorted(
        runs_by_id.values(),
        key=lambda run: (run.get("completed_at", ""), run["run_id"]),
    )


def select_runs(
    runs: list[dict[str, Any]],
    current_id: str | None,
    previous_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(runs) < 2:
        raise ValueError("At least two completed training runs are required.")

    run_indexes = {run["run_id"]: index for index, run in enumerate(runs)}
    if current_id is None:
        current_index = len(runs) - 1
    elif current_id not in run_indexes:
        raise ValueError(f"Completed current run not found: {current_id}")
    else:
        current_index = run_indexes[current_id]

    if previous_id is None:
        if current_index == 0:
            raise ValueError("The selected current run has no previous completed run.")
        previous_index = current_index - 1
    elif previous_id not in run_indexes:
        raise ValueError(f"Completed previous run not found: {previous_id}")
    else:
        previous_index = run_indexes[previous_id]

    if previous_index == current_index:
        raise ValueError("Current and previous run IDs must be different.")
    return runs[previous_index], runs[current_index]


def compare_metrics(
    previous_metrics: dict[str, Any], current_metrics: dict[str, Any]
) -> dict[str, dict[str, float | str]]:
    comparisons: dict[str, dict[str, float | str]] = {}
    shared_metrics = sorted(previous_metrics.keys() & current_metrics.keys())
    for metric in shared_metrics:
        previous_value = previous_metrics[metric]
        current_value = current_metrics[metric]
        if not isinstance(previous_value, (int, float)) or not isinstance(
            current_value, (int, float)
        ):
            continue
        delta = float(current_value) - float(previous_value)
        if delta > METRIC_TOLERANCE:
            direction = "improved"
        elif delta < -METRIC_TOLERANCE:
            direction = "regressed"
        else:
            direction = "unchanged"
        comparisons[metric] = {
            "previous": float(previous_value),
            "current": float(current_value),
            "delta": delta,
            "direction": direction,
        }
    return comparisons


def flatten_settings(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_settings(value, path))
        else:
            flattened[path] = value
    return flattened


def compare_settings(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    previous_settings = flatten_settings(
        {
            "configuration": previous.get("configuration", {}),
            "training_arguments": previous.get("training_arguments", {}),
            "dataset_sizes": previous.get("dataset_sizes", {}),
        }
    )
    current_settings = flatten_settings(
        {
            "configuration": current.get("configuration", {}),
            "training_arguments": current.get("training_arguments", {}),
            "dataset_sizes": current.get("dataset_sizes", {}),
        }
    )
    return {
        key: {
            "previous": previous_settings.get(key),
            "current": current_settings.get(key),
        }
        for key in sorted(previous_settings.keys() | current_settings.keys())
        if previous_settings.get(key) != current_settings.get(key)
    }


def determine_verdict(
    metric_comparisons: dict[str, dict[str, float | str]],
) -> tuple[str, str]:
    for metric in PRIMARY_METRICS:
        comparison = metric_comparisons.get(metric)
        if comparison and comparison["direction"] != "unchanged":
            direction = str(comparison["direction"])
            verdict = "improved" if direction == "improved" else "regressed"
            return verdict, metric
    return "unchanged", PRIMARY_METRICS[0]


def display_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title().replace("F1", "F1")


def format_score(value: float) -> str:
    return f"{value:.4f}"


def format_delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def build_markdown(report: dict[str, Any]) -> str:
    previous = report["previous_run"]
    current = report["current_run"]
    comparisons = report["test_metric_comparisons"]
    verdict = report["verdict"]
    verdict_metric = report["verdict_metric"]
    headline = comparisons.get(verdict_metric)

    if headline:
        conclusion = (
            f"The current run **{verdict}** on test "
            f"{display_metric_name(verdict_metric)} by "
            f"**{format_delta(float(headline['delta']))}**, from "
            f"{format_score(float(headline['previous']))} to "
            f"{format_score(float(headline['current']))}."
        )
    else:
        conclusion = "No shared primary test metric was available for a verdict."

    lines = [
        "# Training Run Comparison",
        "",
        conclusion,
        "",
        "## Runs",
        "",
        "| Role | Run ID | Completed |",
        "| --- | --- | --- |",
        f"| Previous | `{previous['run_id']}` | {previous.get('completed_at', 'Unknown')} |",
        f"| Current | `{current['run_id']}` | {current.get('completed_at', 'Unknown')} |",
        "",
        "## Test Metrics",
        "",
        "| Metric | Previous | Current | Change | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    ordered_metrics = [
        metric for metric in PRIMARY_METRICS if metric in comparisons
    ] + [metric for metric in comparisons if metric not in PRIMARY_METRICS]
    for metric in ordered_metrics:
        comparison = comparisons[metric]
        lines.append(
            f"| {display_metric_name(metric)} "
            f"| {format_score(float(comparison['previous']))} "
            f"| {format_score(float(comparison['current']))} "
            f"| {format_delta(float(comparison['delta']))} "
            f"| {str(comparison['direction']).title()} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Improved metrics: {report['metric_counts']['improved']}",
            f"- Regressed metrics: {report['metric_counts']['regressed']}",
            f"- Unchanged metrics: {report['metric_counts']['unchanged']}",
            f"- Changed settings: {len(report['setting_changes'])}",
        ]
    )

    if report["setting_changes"]:
        lines.extend(
            [
                "",
                "## Setting Changes",
                "",
                "| Setting | Previous | Current |",
                "| --- | --- | --- |",
            ]
        )
        for setting, values in report["setting_changes"].items():
            lines.append(
                f"| `{setting}` | `{values['previous']}` | `{values['current']}` |"
            )

    return "\n".join(lines) + "\n"


def generate_report(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    metric_comparisons = compare_metrics(
        previous.get("test_metrics", {}), current.get("test_metrics", {})
    )
    if not metric_comparisons:
        raise ValueError("The selected runs have no shared numeric test metrics.")

    verdict, verdict_metric = determine_verdict(metric_comparisons)
    metric_counts = {direction: 0 for direction in ("improved", "regressed", "unchanged")}
    for comparison in metric_comparisons.values():
        metric_counts[str(comparison["direction"])] += 1

    return {
        "previous_run": {
            "run_id": previous["run_id"],
            "completed_at": previous.get("completed_at"),
        },
        "current_run": {
            "run_id": current["run_id"],
            "completed_at": current.get("completed_at"),
        },
        "verdict": verdict,
        "verdict_metric": verdict_metric,
        "metric_counts": metric_counts,
        "test_metric_comparisons": metric_comparisons,
        "setting_changes": compare_settings(previous, current),
    }


def main() -> None:
    args = parse_args()
    runs = load_completed_runs(args.history)
    previous, current = select_runs(runs, args.current, args.previous)
    report = generate_report(previous, current)

    report_dir = args.output_dir / f"{previous['run_id']}_vs_{current['run_id']}"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "comparison.json"
    markdown_path = report_dir / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")

    print(build_markdown(report))
    print(f"Reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
