# Training Run Comparison

Compare the two latest completed training runs:

```bash
python comparison/compare_runs.py
```

Reports are written to:

```text
comparison/results/<previous-run-id>_vs_<current-run-id>/
├── comparison.md
└── comparison.json
```

The Markdown report gives a final improved, regressed, or unchanged verdict based primarily on held-out test macro F1, with test micro F1 as the tie-breaker. It also lists every shared test metric and changed training setting. The JSON report contains the same data for automation.

Compare specific completed runs:

```bash
python comparison/compare_runs.py --previous <run-id> --current <run-id>
```

Use a different history or report directory:

```bash
python comparison/compare_runs.py --history <path-to-training_history.jsonl> --output-dir <directory>
```

At least two completed runs must exist in `outputs/bert-movie-genres/training_history.jsonl`. Failed and interrupted runs remain in training history but are excluded from comparisons.
