import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from model_input import MAX_LENGTH, build_model_text


MODEL_NAME = "google-bert/bert-base-uncased"
DATASET_PATH = Path(__file__).parent / "data" / "cleaned_movies.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "bert-movie-genres"
RUNS_DIR = OUTPUT_DIR / "runs"
HISTORY_PATH = OUTPUT_DIR / "training_history.jsonl"
LABEL_THRESHOLD = 0.5
EVAL_STEPS = 2500
THRESHOLD_CANDIDATES = np.arange(0.1, 0.91, 0.05)
LOSS_TYPE = "weighted_bce"  # Use "focal" for weighted focal loss.
FOCAL_GAMMA = 2.0
MAX_POS_WEIGHT = 10.0
USE_POS_WEIGHTS = True
OVERSAMPLE_RATIO = 0.0
OVERSAMPLE_POWER = 0.5
RANDOM_SEED = 42

CLASS_LABELS = [
    "Adventure",
    "War",
    "Documentary",
    "Science Fiction",
    "Thriller",
    "Family",
    "Romance",
    "Fantasy",
    "Drama",
    "Action",
    "Music",
    "Western",
    "Animation",
    "Mystery",
    "TV Movie",
    "Horror",
    "Comedy",
    "Crime",
    "History",
]
LABEL_TO_ID = {label: index for index, label in enumerate(CLASS_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data, *, indent=2) -> None:
    path.write_text(
        json.dumps(data, indent=indent, default=json_default),
        encoding="utf-8",
    )


def archive_run_summary(run_dir: Path, summary: dict) -> None:
    write_json(run_dir / "run_summary.json", summary)
    with HISTORY_PATH.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(summary, default=json_default) + "\n")
    shutil.copy2(run_dir / "run_summary.json", OUTPUT_DIR / "latest_run.json")


def label_metric_name(label: str) -> str:
    return label.lower().replace(" ", "_")


def multilabel_f1(predictions, targets):
    true_positives = np.logical_and(predictions, targets).sum(axis=0)
    false_positives = np.logical_and(predictions, ~targets).sum(axis=0)
    false_negatives = np.logical_and(~predictions, targets).sum(axis=0)

    per_label_denominator = 2 * true_positives + false_positives + false_negatives
    per_label_f1 = np.divide(
        2 * true_positives,
        per_label_denominator,
        out=np.zeros_like(true_positives, dtype=float),
        where=per_label_denominator != 0,
    )
    micro_denominator = (
        2 * true_positives.sum() + false_positives.sum() + false_negatives.sum()
    )
    micro_f1 = 2 * true_positives.sum() / max(micro_denominator, 1)
    return true_positives, false_positives, false_negatives, per_label_f1, micro_f1


def average_precision(probabilities, targets) -> float:
    positive_count = targets.sum()
    if positive_count == 0:
        return 0.0

    sorted_targets = targets[np.argsort(-probabilities)]
    precision_at_rank = np.cumsum(sorted_targets) / np.arange(
        1, len(sorted_targets) + 1
    )
    return float(precision_at_rank[sorted_targets].sum() / positive_count)


def encode_batch(examples, tokenizer):
    texts = [
        build_model_text(title, overview, keywords)
        for title, overview, keywords in zip(
            examples["title"],
            examples["overview"],
            examples["keywords"],
        )
    ]
    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
    )
    labels = []
    for genre_names in examples["genre_names"]:
        multi_hot = [0.0] * len(CLASS_LABELS)
        for genre_name in genre_names.split(","):
            genre_name = genre_name.strip()
            if genre_name not in LABEL_TO_ID:
                raise ValueError(f"Unknown genre in dataset: {genre_name}")
            multi_hot[LABEL_TO_ID[genre_name]] = 1.0
        labels.append(multi_hot)
    encoded["labels"] = labels
    return encoded


def find_per_label_thresholds(probabilities, targets) -> np.ndarray:
    per_label_thresholds = []
    for index in range(len(CLASS_LABELS)):
        label_scores = []
        for threshold in THRESHOLD_CANDIDATES:
            *_, per_label_f1, _ = multilabel_f1(
                probabilities[:, [index]] >= threshold,
                targets[:, [index]],
            )
            label_scores.append(per_label_f1[0])
        per_label_thresholds.append(
            float(THRESHOLD_CANDIDATES[np.argmax(label_scores)])
        )
    return np.asarray(per_label_thresholds)


def compute_metrics(eval_prediction, thresholds=None):
    logits, labels = eval_prediction
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    targets = labels >= 0.5
    if thresholds is None:
        thresholds = find_per_label_thresholds(probabilities, targets)
    predictions = probabilities >= thresholds

    (
        true_positives,
        false_positives,
        false_negatives,
        per_label_f1,
        micro_f1,
    ) = multilabel_f1(predictions, targets)
    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_positives) != 0,
    )
    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_negatives) != 0,
    )

    per_label_average_precision = np.asarray(
        [
            average_precision(probabilities[:, index], targets[:, index])
            for index in range(len(CLASS_LABELS))
        ]
    )
    metrics = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(per_label_f1.mean()),
        "macro_average_precision": float(per_label_average_precision.mean()),
    }
    for index, label in ID_TO_LABEL.items():
        metric_name = label_metric_name(label)
        metrics[f"{metric_name}_precision"] = float(precision[index])
        metrics[f"{metric_name}_recall"] = float(recall[index])
        metrics[f"{metric_name}_f1"] = float(per_label_f1[index])
        metrics[f"{metric_name}_average_precision"] = float(
            per_label_average_precision[index]
        )
    return metrics


def calculate_pos_weights(train_dataset) -> np.ndarray:
    labels = np.asarray(train_dataset["labels"], dtype=np.float32)
    positive_counts = labels.sum(axis=0)
    if np.any(positive_counts == 0):
        missing_labels = [
            CLASS_LABELS[index]
            for index in np.flatnonzero(positive_counts == 0)
        ]
        raise ValueError(f"Training split has no examples for: {missing_labels}")

    if not USE_POS_WEIGHTS:
        return np.ones(len(CLASS_LABELS), dtype=np.float32)

    negative_counts = len(labels) - positive_counts
    return np.clip(negative_counts / positive_counts, 1.0, MAX_POS_WEIGHT)


def predict_without_metrics(trainer, dataset):
    compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    try:
        return trainer.predict(dataset)
    finally:
        trainer.compute_metrics = compute_metrics


def oversample_minority_examples(train_dataset):
    if OVERSAMPLE_RATIO <= 0:
        return train_dataset

    labels = np.asarray(train_dataset["labels"], dtype=np.float32)
    prevalence = labels.mean(axis=0)
    label_weights = np.power(
        np.maximum(prevalence, 1 / len(labels)), -OVERSAMPLE_POWER
    )
    sample_weights = np.max(labels * label_weights, axis=1)
    sample_probabilities = sample_weights / sample_weights.sum()

    additional_count = round(len(train_dataset) * OVERSAMPLE_RATIO)
    random_generator = np.random.default_rng(RANDOM_SEED)
    extra_indices = random_generator.choice(
        len(train_dataset),
        size=additional_count,
        replace=True,
        p=sample_probabilities,
    )
    all_indices = np.concatenate((np.arange(len(train_dataset)), extra_indices))
    random_generator.shuffle(all_indices)
    return train_dataset.select(all_indices.tolist())


class ImbalanceAwareTrainer(Trainer):
    def __init__(self, *args, pos_weights, loss_type=LOSS_TYPE, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_type not in {"weighted_bce", "focal"}:
            raise ValueError("loss_type must be 'weighted_bce' or 'focal'")
        self.pos_weights = torch.as_tensor(pos_weights, dtype=torch.float32)
        self.loss_type = loss_type

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weights.to(logits.device, dtype=logits.dtype),
            reduction="none",
        )
        if self.loss_type == "focal":
            probabilities = torch.sigmoid(logits)
            correct_class_probabilities = torch.where(
                labels.bool(), probabilities, 1 - probabilities
            )
            element_loss *= (1 - correct_class_probabilities).pow(FOCAL_GAMMA)
        loss = element_loss.mean()
        return (loss, outputs) if return_outputs else loss


def tune_thresholds(prediction_output, output_dir: Path) -> np.ndarray:
    probabilities = 1.0 / (1.0 + np.exp(-prediction_output.predictions))
    targets = prediction_output.label_ids >= 0.5
    per_label_thresholds = find_per_label_thresholds(probabilities, targets)

    optimized_predictions = probabilities >= per_label_thresholds
    *_, optimized_per_label_f1, optimized_micro_f1 = multilabel_f1(
        optimized_predictions, targets
    )
    threshold_data = {
        "per_label_thresholds": {
            label: round(float(threshold), 2)
            for label, threshold in zip(CLASS_LABELS, per_label_thresholds)
        },
        "validation_micro_f1": float(optimized_micro_f1),
        "validation_macro_f1": float(optimized_per_label_f1.mean()),
    }
    threshold_path = output_dir / "thresholds.json"
    threshold_path.write_text(json.dumps(threshold_data, indent=2), encoding="utf-8")

    _, axis = plt.subplots(figsize=(10, 8))
    label_positions = np.arange(len(CLASS_LABELS))
    axis.barh(label_positions, per_label_thresholds)
    axis.axvline(LABEL_THRESHOLD, color="tab:red", linestyle="--", label="Default")
    axis.set_yticks(label_positions, CLASS_LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_title("Tuned Threshold by Genre")
    axis.set_xlabel("Threshold")
    axis.legend()
    axis.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    threshold_graph_path = output_dir / "threshold_tuning.png"
    plt.savefig(threshold_graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Tuned thresholds saved to: {threshold_path}")
    print(f"Threshold tuning graph saved to: {threshold_graph_path}")
    return per_label_thresholds


def plot_training_metrics(log_history, output_dir: Path) -> None:
    """Plot and save metrics collected by Trainer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_groups = [
        ("loss", "Training Loss"),
        ("eval_loss", "Validation Loss"),   
        ("eval_micro_f1", "Validation Micro F1"),
        ("eval_macro_f1", "Validation Macro F1"),
        ("eval_macro_average_precision", "Validation Macro Average Precision"),
    ]

    _, axes = plt.subplots(3, 2, figsize=(12, 12))

    for axis, (metric, title) in zip(axes.flat, metric_groups):
        records = [entry for entry in log_history if metric in entry]
        x_values = [entry.get("epoch", entry.get("step", 0)) for entry in records]
        y_values = [entry[metric] for entry in records]

        if records:
            axis.plot(x_values, y_values, marker="o")
        else:
            axis.text(0.5, 0.5, "No data collected", ha="center", va="center")

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.3)

    for axis in axes.flat[len(metric_groups):]:
        axis.axis("off")

    plt.tight_layout()
    graph_path = output_dir / "training_metrics.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training graphs saved to: {graph_path}")


def plot_per_label_metrics(metrics, output_dir: Path) -> None:
    metric_groups = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("average_precision", "Average Precision"),
    ]
    label_positions = np.arange(len(CLASS_LABELS))
    bar_height = 0.2

    _, axis = plt.subplots(figsize=(14, 10))
    for offset_index, (metric, title) in enumerate(metric_groups):
        values = [
            metrics[f"{label_metric_name(label)}_{metric}"]
            for label in CLASS_LABELS
        ]
        offset = (offset_index - 1.5) * bar_height
        axis.barh(label_positions + offset, values, bar_height, label=title)

    axis.set_yticks(label_positions, CLASS_LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_title("Per-Genre Metrics for Best Checkpoint")
    axis.set_xlabel("Score")
    axis.legend(ncols=2)
    axis.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    graph_path = output_dir / "per_label_metrics.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-label metrics graph saved to: {graph_path}")


def plot_per_label_history(log_history, output_dir: Path) -> None:
    evaluation_records = [
        entry for entry in log_history if "eval_macro_f1" in entry
    ]
    if not evaluation_records:
        print("No per-label evaluation history available to plot.")
        return

    _, axes = plt.subplots(2, 1, figsize=(16, 14), constrained_layout=True)
    history_metrics = [("f1", "F1"), ("average_precision", "Average Precision")]
    steps = [entry["step"] for entry in evaluation_records]
    tick_stride = max(1, len(steps) // 10)
    tick_positions = np.arange(0, len(steps), tick_stride)

    for axis, (metric, title) in zip(axes, history_metrics):
        values = np.asarray(
            [
                [
                    entry[f"eval_{label_metric_name(label)}_{metric}"]
                    for entry in evaluation_records
                ]
                for label in CLASS_LABELS
            ]
        )
        image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axis.set_yticks(np.arange(len(CLASS_LABELS)), CLASS_LABELS)
        axis.set_xticks(tick_positions, [steps[index] for index in tick_positions])
        axis.set_title(f"Per-Genre {title} During Training")
        axis.set_xlabel("Training Step")
        axis.set_ylabel("Genre")
        plt.colorbar(image, ax=axis, label=title)

    graph_path = output_dir / "per_label_metric_history.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-label metric history saved to: {graph_path}")


def train_run(run_id: str, started_at: datetime, run_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        problem_type="multi_label_classification",
    )

    full_dataset = load_dataset("csv", data_files=str(DATASET_PATH), split="train")
    dataset = DatasetDict(
        {
            split: full_dataset.filter(
                lambda example, split=split: example["split"] == split,
                num_proc=2,
            )
            for split in ["train", "validation", "test"]
        }
    )
    dataset = dataset.map(
        encode_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
        num_proc=2,
    )
    pos_weights = calculate_pos_weights(dataset["train"])
    original_train_size = len(dataset["train"])
    dataset["train"] = oversample_minority_examples(dataset["train"])
    print(
        f"Training examples: {original_train_size:,} -> "
        f"{len(dataset['train']):,} after minority oversampling"
    )
    print(
        "Positive class weights: "
        + ", ".join(
            f"{label}={weight:.2f}"
            for label, weight in zip(CLASS_LABELS, pos_weights)
        )
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="macro_average_precision",
        greater_is_better=True,
        fp16=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    trainer = ImbalanceAwareTrainer(
        model=model,
        pos_weights=pos_weights,
        loss_type=LOSS_TYPE,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=training_args,
    )
    train_result = trainer.train()
    validation_predictions = trainer.predict(
        dataset["validation"], metric_key_prefix="validation"
    )
    thresholds = tune_thresholds(validation_predictions, run_dir)
    test_predictions = predict_without_metrics(trainer, dataset["test"])
    test_metrics = compute_metrics(
        (test_predictions.predictions, test_predictions.label_ids),
        thresholds=thresholds,
    )
    write_json(run_dir / "test_metrics.json", test_metrics)
    write_json(run_dir / "trainer_log_history.json", trainer.state.log_history)
    plot_training_metrics(trainer.state.log_history, run_dir)
    plot_per_label_metrics(test_metrics, run_dir)
    plot_per_label_history(trainer.state.log_history, run_dir)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    latest_artifacts = [
        "thresholds.json",
        "test_metrics.json",
        "training_metrics.png",
        "per_label_metrics.png",
        "per_label_metric_history.png",
        "threshold_tuning.png",
    ]
    for artifact_name in latest_artifacts:
        artifact_path = run_dir / artifact_name
        if artifact_path.exists():
            shutil.copy2(artifact_path, OUTPUT_DIR / artifact_name)

    threshold_metrics = json.loads(
        (run_dir / "thresholds.json").read_text(encoding="utf-8")
    )
    summary = {
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "configuration": {
            "max_length": MAX_LENGTH,
            "loss_type": LOSS_TYPE,
            "focal_gamma": FOCAL_GAMMA,
            "max_pos_weight": MAX_POS_WEIGHT,
            "use_pos_weights": USE_POS_WEIGHTS,
            "oversample_ratio": OVERSAMPLE_RATIO,
            "oversample_power": OVERSAMPLE_POWER,
            "random_seed": RANDOM_SEED,
        },
        "training_arguments": training_args.to_dict(),
        "dataset_sizes": {
            "train_before_oversampling": original_train_size,
            "train_after_oversampling": len(dataset["train"]),
            "validation": len(dataset["validation"]),
            "test": len(dataset["test"]),
        },
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_predictions.metrics,
        "threshold_metrics": threshold_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }
    archive_run_summary(run_dir, summary)
    print(f"Training history archived to: {run_dir}")


def main() -> None:
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "run_summary.json",
        {
            "run_id": run_id,
            "status": "running",
            "started_at": started_at.isoformat(),
            "model_name": MODEL_NAME,
        },
    )

    try:
        train_run(run_id, started_at, run_dir)
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        archive_run_summary(
            run_dir,
            {
                "run_id": run_id,
                "status": status,
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model_name": MODEL_NAME,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
