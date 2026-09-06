import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "outputs" / "bert-movie-genres"
HISTORY_PATH = MODEL_DIR / "training_history.jsonl"
DIST_DIR = Path(__file__).resolve().parent / "dist"
RELEASE_ASSETS = (
    "latest_run.json",
    "test_metrics.json",
    "thresholds.json",
    "training_metrics.png",
    "per_label_metrics.png",
    "per_label_metric_history.png",
    "threshold_tuning.png",
)


def restore_history(repo_id: str, token: str | None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename="training_history.jsonl",
            repo_type="model",
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        print("No previous training history found; starting with the first run.")
        return
    shutil.copy2(downloaded_path, HISTORY_PATH)
    print(f"Restored training history to: {HISTORY_PATH}")


def completed_run_count() -> int:
    if not HISTORY_PATH.is_file():
        return 0
    return sum(
        1
        for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("status") == "completed"
    )


def prepare_release() -> None:
    latest_path = MODEL_DIR / "latest_run.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Successful run summary not found: {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("status") != "completed":
        raise ValueError("The latest training run did not complete successfully.")

    run_id = latest["run_id"]
    tag = f"model-{run_id}"
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    for artifact_name in RELEASE_ASSETS:
        source = MODEL_DIR / artifact_name
        if source.is_file():
            shutil.copy2(source, DIST_DIR / artifact_name)

    comparison_path = None
    comparison_output_dir = DIST_DIR / "comparisons"
    if completed_run_count() >= 2:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "comparison" / "compare_runs.py"),
                "--history",
                str(HISTORY_PATH),
                "--output-dir",
                str(comparison_output_dir),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        comparison_candidates = sorted(
            comparison_output_dir.glob(f"*_vs_{run_id}/comparison.md")
        )
        if comparison_candidates:
            comparison_path = comparison_candidates[-1]
            shutil.copy2(comparison_path, DIST_DIR / "comparison.md")
            shutil.copy2(
                comparison_path.with_suffix(".json"),
                DIST_DIR / "comparison.json",
            )

    macro_f1 = latest.get("test_metrics", {}).get("macro_f1")
    micro_f1 = latest.get("test_metrics", {}).get("micro_f1")
    notes = [
        f"# Model training {run_id}",
        "",
        f"- Base model: `{latest.get('model_name', 'unknown')}`",
        f"- Completed: {latest.get('completed_at', 'unknown')}",
        f"- Test macro F1: {macro_f1:.4f}" if macro_f1 is not None else "- Test macro F1: unavailable",
        f"- Test micro F1: {micro_f1:.4f}" if micro_f1 is not None else "- Test micro F1: unavailable",
    ]
    if comparison_path:
        notes.extend(["", comparison_path.read_text(encoding="utf-8")])
    else:
        notes.extend(["", "No previous completed run is available for comparison."])
    (DIST_DIR / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    if comparison_output_dir.exists():
        shutil.rmtree(comparison_output_dir)

    metadata = {"run_id": run_id, "tag": tag, "dist_dir": str(DIST_DIR)}
    (DIST_DIR / "release.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata))


def tag_hugging_face_release(repo_id: str, tag: str, token: str | None) -> None:
    HfApi(token=token).create_tag(
        repo_id=repo_id,
        tag=tag,
        repo_type="model",
        tag_message=f"Successful training release {tag}",
    )
    print(f"Created Hugging Face tag: {tag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage automated model releases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    restore_parser = subparsers.add_parser("restore-history")
    restore_parser.add_argument("repo_id")
    restore_parser.add_argument("--token")

    subparsers.add_parser("prepare")

    tag_parser = subparsers.add_parser("tag-hugging-face")
    tag_parser.add_argument("repo_id")
    tag_parser.add_argument("tag")
    tag_parser.add_argument("--token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "restore-history":
        restore_history(args.repo_id, args.token)
    elif args.command == "prepare":
        prepare_release()
    elif args.command == "tag-hugging-face":
        tag_hugging_face_release(args.repo_id, args.tag, args.token)


if __name__ == "__main__":
    main()