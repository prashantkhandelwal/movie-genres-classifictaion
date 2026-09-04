from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "google-bert/bert-base-uncased"
DATASET_PATH = Path(__file__).parent / "data" / "cleaned_movies.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "bert-movie-genres"
MAX_LENGTH = 256
LABEL_THRESHOLD = 0.5

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


def encode_batch(examples, tokenizer):
    encoded = tokenizer(
        examples["plot"],
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


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = probabilities >= LABEL_THRESHOLD
    targets = labels >= 0.5

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

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(per_label_f1.mean()),
    }


def plot_training_metrics(log_history, output_dir: Path) -> None:
    """Plot and save metrics collected by Trainer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_groups = [
        ("loss", "Training Loss"),
        ("eval_loss", "Validation Loss"),
        ("eval_micro_f1", "Validation Micro F1"),
        ("eval_macro_f1", "Validation Macro F1"),
    ]

    _, axes = plt.subplots(2, 2, figsize=(12, 8))

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

    plt.tight_layout()
    graph_path = output_dir / "training_metrics.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Training graphs saved to: {graph_path}")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        problem_type="multi_label_classification",
    )

    dataset = load_dataset("csv", data_files=str(DATASET_PATH), split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=3407)
    dataset = dataset.map(
        encode_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
        num_proc=2,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            fp16=True,
            logging_steps=100,
            save_total_limit=2,
            report_to="none",
        ),
    )
    trainer.train()
    plot_training_metrics(trainer.state.log_history, OUTPUT_DIR)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
