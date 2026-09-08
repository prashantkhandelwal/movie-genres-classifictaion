from collections.abc import Callable, Sequence

import numpy as np


DEFAULT_THRESHOLD_CANDIDATES = np.arange(0.1, 0.91, 0.05)


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


def find_per_label_thresholds(
    probabilities,
    targets,
    threshold_candidates=DEFAULT_THRESHOLD_CANDIDATES,
) -> np.ndarray:
    threshold_candidates = np.asarray(threshold_candidates, dtype=float)
    if threshold_candidates.ndim != 1 or threshold_candidates.size == 0:
        raise ValueError("threshold_candidates must be a non-empty one-dimensional array")

    per_label_thresholds = []
    for index in range(probabilities.shape[1]):
        label_scores = []
        for threshold in threshold_candidates:
            *_, per_label_f1, _ = multilabel_f1(
                probabilities[:, [index]] >= threshold,
                targets[:, [index]],
            )
            label_scores.append(per_label_f1[0])
        per_label_thresholds.append(
            float(threshold_candidates[np.argmax(label_scores)])
        )
    return np.asarray(per_label_thresholds)


def compute_multilabel_metrics(
    eval_prediction,
    labels: Sequence[str],
    thresholds=None,
    threshold_candidates=DEFAULT_THRESHOLD_CANDIDATES,
) -> dict[str, float]:
    logits, target_values = eval_prediction
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    targets = target_values >= 0.5
    if probabilities.ndim != 2 or probabilities.shape != targets.shape:
        raise ValueError("logits and targets must be equally shaped two-dimensional arrays")
    if probabilities.shape[1] != len(labels):
        raise ValueError("label count must match the second dimension of logits")

    if thresholds is None:
        thresholds = find_per_label_thresholds(
            probabilities,
            targets,
            threshold_candidates,
        )
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
            for index in range(len(labels))
        ]
    )

    metrics = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(per_label_f1.mean()),
        "macro_average_precision": float(per_label_average_precision.mean()),
    }
    for index, label in enumerate(labels):
        metric_name = label_metric_name(label)
        metrics[f"{metric_name}_precision"] = float(precision[index])
        metrics[f"{metric_name}_recall"] = float(recall[index])
        metrics[f"{metric_name}_f1"] = float(per_label_f1[index])
        metrics[f"{metric_name}_average_precision"] = float(
            per_label_average_precision[index]
        )
    return metrics


def make_multilabel_compute_metrics(
    labels: Sequence[str],
    *,
    threshold_candidates=DEFAULT_THRESHOLD_CANDIDATES,
) -> Callable:
    labels = tuple(labels)
    threshold_candidates = np.asarray(threshold_candidates, dtype=float)

    def compute_metrics(eval_prediction, thresholds=None) -> dict[str, float]:
        return compute_multilabel_metrics(
            eval_prediction,
            labels,
            thresholds,
            threshold_candidates,
        )

    return compute_metrics