# Movie Genre Classification with BERT

This model predicts one or more movie genres from a movie's plot overview. It is a multi-label text-classification model built by fine-tuning `google-bert/bert-base-uncased` with Hugging Face Transformers.

For each input movie, the model produces an independent score for every genre. A movie can therefore be classified as both `Action` and `Science Fiction`, for example.

## Quick Start

The project requires Python 3.12 or newer. Install the locked project dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

The repository includes a trained model under `outputs/bert-movie-genres`. Predict genres from a plot overview with:

```bash
uv run python test.py "A crew travels through deep space to stop an alien threat from destroying Earth."
```

The command uses CUDA when available and applies the validation-tuned threshold for each genre from `outputs/bert-movie-genres/thresholds.json`. To replace those thresholds with one value:

```bash
uv run python test.py --threshold 0.5 "A crew travels through deep space to stop an alien threat."
```

Omit the overview to enter it interactively.

## Project Workflow

Prepare the dataset, train the model, and run inference:

```bash
uv run python data.py
uv run python validation.py
uv run python train.py
uv run python test.py "A detective investigates a series of unexplained disappearances."
```

`data.py` reads `data/moviedb.movies.csv` and `data/genres.csv`, then writes the cleaned and split dataset to `data/cleaned_movies.csv`. `validation.py` checks its schema, split isolation, labels, and basic quality constraints. Training requires a CUDA-capable device because FP16 is enabled in `train.py`.

Run the focused test suite with:

```bash
uv run python -m unittest discover -s tests
```

## Model Details

- **Model type:** BERT base uncased with a sequence-classification head
- **Task:** Multi-label text classification
- **Base model:** [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)
- **Number of labels:** 19
- **Maximum input length during training:** 256 tokens
- **Classification thresholds:** tuned independently per genre on validation data
- **Framework:** Hugging Face Transformers 5.5.0 and PyTorch

## Intended Use

This model is intended for experimentation, movie catalog organization, genre-based search, recommendation prototypes, and educational NLP projects. It predicts genres from textual movie metadata; it does not understand the full movie, watch its content, or provide content ratings.

## Labels

The model predicts the following genres:

`Adventure`, `War`, `Documentary`, `Science Fiction`, `Thriller`, `Family`, `Romance`, `Fantasy`, `Drama`, `Action`, `Music`, `Western`, `Animation`, `Mystery`, `TV Movie`, `Horror`, `Comedy`, `Crime`, and `History`.

## Input Format

Provide a movie plot overview. Inputs are lowercased by the uncased BERT tokenizer and truncated to 256 tokens during training and inference.

## Standalone Usage

The published model can be used in another project without cloning this repository:

```bash
pip install torch transformers
```

```python
from transformers import pipeline

classifier = pipeline(
	"text-classification",
	model="prashantkhandelwal/movie-genres-classification",
	top_k=None,
	function_to_apply="sigmoid",
)

overview = "A crew travels through deep space to stop an alien threat."
scores = classifier(f"Overview: {overview}", truncation=True, max_length=256)
genres = [result for result in scores if result["score"] >= 0.5]
print(genres)
```

The standalone example uses a fixed `0.5` threshold. The local `test.py` command uses tuned per-genre thresholds when available. A threshold is a decision rule, not a calibrated probability guarantee; applications should choose thresholds using validation data and their preferred precision/recall tradeoff.

## Training

The model was fine-tuned with the following configuration:

- Dataset split: 80% training, 10% validation, and 10% held-out testing
- Split assignment: deterministic iterative multi-label stratification, preserving per-genre prevalence
- Epochs: `3`
- Learning rate: `2e-5`
- Training batch size per device: `16`
- Evaluation batch size per device: `32`
- Weight decay: `0.01`
- Warmup ratio: `0.1`
- Evaluation and checkpoint saving: every 2,500 steps
- Best checkpoint metric: validation macro average precision
- Early stopping patience: 3 evaluations
- Mixed precision: FP16 when supported by the training hardware
- Loss: positive-class-weighted BCE, with weights capped at `10.0`
- Oversampling: disabled by default; set `OVERSAMPLE_RATIO` above `0` to enable it

Genre names are converted to multi-hot vectors. The training objective treats each genre as an independent binary decision. Positive-class weighting is enabled independently with `USE_POS_WEIGHTS`; oversampling is controlled by `OVERSAMPLE_RATIO`, which defaults to `0` so the two imbalance treatments are not compounded. Set `LOSS_TYPE` to `weighted_bce` or `focal`; `FOCAL_GAMMA`, `MAX_POS_WEIGHT`, and `OVERSAMPLE_POWER` provide the remaining controls.

Evaluation reports micro F1, macro F1, macro average precision, and per-genre precision, recall, F1, and average precision. Threshold-independent macro average precision selects the best checkpoint. Thresholds are then optimized independently for every genre on the validation split and saved to `thresholds.json`. Raw held-out test predictions are generated without running the Trainer metric callback, then evaluated once using the validation threshold vector and saved to `test_metrics.json`.

Training saves the following visualizations in the model output directory:

- `training_metrics.png`: training loss, validation loss, micro F1, macro F1, and macro average precision
- `per_label_metrics.png`: precision, recall, F1, and average precision for each genre at the best checkpoint
- `per_label_metric_history.png`: per-genre F1 and average-precision heatmaps across evaluation steps
- `threshold_tuning.png`: global threshold search and tuned threshold for each genre

Each training attempt is assigned a UTC run ID under `outputs/bert-movie-genres/runs/`. Completed runs retain their configuration, dataset sizes, Trainer log history, validation and test metrics, tuned thresholds, and plots. Failed or interrupted runs retain their status and error. `training_history.jsonl` provides one summary record per finished attempt for comparison, while `latest_run.json` and the artifacts in the model output directory reflect the latest attempt and latest successful model outputs respectively. Model weights are kept only in the main output directory to avoid duplicating large files for every run.

After at least two runs complete, generate a comparison with `python comparison/compare_runs.py`. See [`comparison/README.md`](comparison/README.md) for explicit run selection and output details.

Successful training can also publish a versioned model to Hugging Face and create a matching GitHub Release through GitHub Actions. See [`release/README.md`](release/README.md) for runner requirements, repository secrets, and usage.

## Evaluation

The training script computes:

- **Micro F1:** pools true positives, false positives, and false negatives across all genres before calculating F1.
- **Macro F1:** calculates F1 independently for each genre and averages the results.
- **Macro average precision:** averages threshold-independent average precision across genres.
- **Per-genre metrics:** records precision, recall, F1, average precision, support, and tuned thresholds for each label.

The latest held-out scores are stored in `outputs/bert-movie-genres/test_metrics.json`. Run-specific results are archived under `outputs/bert-movie-genres/runs/<run-id>/`; use the comparison tool to evaluate changes between completed runs rather than relying on training loss alone.

## Utilities

Compare the two latest completed training runs:

```bash
uv run python comparison/compare_runs.py
```

Profile the raw source dataset or validate the cleaned dataset:

```bash
uv run python movie_dataset_profile.py
uv run python validation.py
```

Upload the latest model files to a Hugging Face model repository after setting `HF_REPO_ID` and `HF_TOKEN` in `.env`:

```bash
uv run python hf/hf_upload.py
```

See [`comparison/README.md`](comparison/README.md) for run selection and [`release/README.md`](release/README.md) for the automated GitHub Actions release workflow.

## Repository Layout

```text
data.py                  Clean and split the source dataset
train.py                 Train, evaluate, tune thresholds, and archive a run
test.py                  Run local inference
metrics.py               Multi-label metrics and threshold tuning
model_input.py           Shared inference and training input format
validation.py            Validate the cleaned dataset
comparison/              Compare completed training runs
release/                 Build automated model releases
outputs/bert-movie-genres/
						 Latest model artifacts and run history
tests/                   Focused unit tests
```

## Limitations and Biases

- Predictions depend on the quality, language, completeness, and style of the movie overview.
- The model was trained on movie metadata and may reproduce genre-labeling patterns or omissions in that data.
- Rare genres may receive less reliable predictions than common genres; inspect macro F1 and per-genre results.
- Genre-specific thresholds can drift when the training data distribution changes and should be retuned after training.
- Text longer than 256 tokens is truncated, which can remove useful plot information.
- The model is based on English BERT and should not be assumed to perform reliably on other languages.
- This model has not been validated for safety-critical, legal, or high-impact decision-making.

## Dataset and Provenance

The training pipeline uses the project's `cleaned_movies.csv`. Its model input contains only the overview, while genres remain comma-separated multi-label targets. The cleaner normalizes whitespace, removes missing, placeholder, and undersized overviews, merges duplicate overview records, rejects records with more than six genres, and uses seeded iterative multi-label stratification for leakage-resistant train, validation, and test splits with similar genre prevalence. The repository includes database queries that extract movie metadata and genres, but the exact upstream dataset version and license are not recorded in the training script. Users should verify the source data's terms of use before redistributing or deploying the model.

## Files

The inference model directory contains the model weights and tokenizer files together:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `thresholds.json` (validation-tuned per-genre decision thresholds)
- `test_metrics.json` (held-out evaluation metrics)

Checkpoint optimizer and random-state files are not required for inference.

## License

The repository code is licensed under the terms in [`LICENSE`](LICENSE). The model and source dataset may have separate terms; review the licenses of the base model and source movie metadata before using or redistributing them.
