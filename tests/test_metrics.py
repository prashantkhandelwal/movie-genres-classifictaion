import unittest

import numpy as np

from metrics import compute_multilabel_metrics, make_multilabel_compute_metrics


class MultilabelMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_prediction = (
            np.asarray([[10.0, -10.0], [-10.0, 10.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )

    def test_callback_supports_custom_labels_and_candidates(self) -> None:
        compute_metrics = make_multilabel_compute_metrics(
            ["Alpha Label", "Beta"],
            threshold_candidates=[0.5],
        )

        metrics = compute_metrics(self.eval_prediction)

        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["macro_average_precision"], 1.0)
        self.assertEqual(metrics["alpha_label_precision"], 1.0)
        self.assertEqual(metrics["beta_recall"], 1.0)

    def test_explicit_thresholds_can_be_reused_on_another_split(self) -> None:
        metrics = compute_multilabel_metrics(
            self.eval_prediction,
            ["Alpha", "Beta"],
            thresholds=np.asarray([0.5, 0.5]),
        )

        self.assertEqual(metrics["alpha_f1"], 1.0)
        self.assertEqual(metrics["beta_average_precision"], 1.0)

    def test_label_count_must_match_model_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "label count"):
            compute_multilabel_metrics(self.eval_prediction, ["Alpha"])


if __name__ == "__main__":
    unittest.main()