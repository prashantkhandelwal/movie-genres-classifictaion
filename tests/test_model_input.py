import unittest
from unittest.mock import patch

import test
from model_input import build_model_text


class ModelInputTests(unittest.TestCase):
    def test_model_formatter_only_includes_overview(self) -> None:
        self.assertEqual(
            build_model_text("Ocean voyage"),
            "Overview: Ocean voyage",
        )

    def test_cli_only_prompts_for_overview(self) -> None:
        captured = {}

        def fake_predict(overview, threshold=None):
            captured.update(
                overview=overview,
                threshold=threshold,
            )
            return []

        with (
            patch("sys.argv", ["test.py"]),
            patch(
                "builtins.input",
                side_effect=["Ocean voyage"],
            ),
            patch.object(test, "predict_genres", fake_predict),
        ):
            test.main()

        self.assertEqual(
            captured,
            {"overview": "Ocean voyage", "threshold": None},
        )


if __name__ == "__main__":
    unittest.main()