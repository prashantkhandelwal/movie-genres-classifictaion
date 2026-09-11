import unittest

import numpy as np

from data import iterative_multilabel_splits


class IterativeMultilabelSplitTests(unittest.TestCase):
    def test_split_sizes_and_label_distribution(self) -> None:
        label_matrix = np.zeros((100, 3), dtype=np.int8)
        label_matrix[:, 0] = 1
        label_matrix[::2, 1] = 1
        label_matrix[::5, 2] = 1

        splits = iterative_multilabel_splits(label_matrix)

        self.assertEqual(np.count_nonzero(splits == "train"), 80)
        self.assertEqual(np.count_nonzero(splits == "validation"), 10)
        self.assertEqual(np.count_nonzero(splits == "test"), 10)
        overall_prevalence = label_matrix.mean(axis=0)
        for split_name in ("train", "validation", "test"):
            split_prevalence = label_matrix[splits == split_name].mean(axis=0)
            np.testing.assert_allclose(
                split_prevalence,
                overall_prevalence,
                atol=0.05,
            )

    def test_split_is_deterministic(self) -> None:
        label_matrix = np.tile(
            np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int8),
            (10, 1),
        )

        first = iterative_multilabel_splits(label_matrix, random_seed=7)
        second = iterative_multilabel_splits(label_matrix, random_seed=7)

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()