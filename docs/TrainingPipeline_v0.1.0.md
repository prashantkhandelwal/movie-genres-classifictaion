# Movie Genre Training Pipeline

**Documentation version:** `0.1.0`  
**Project version:** `0.1.0`  
**Documented entry point:** `train.py`  
**Last verified:** 2026-09-06

## 1. Purpose and Scope

The training pipeline fine-tunes `google-bert/bert-base-uncased` to infer one or
more genres from a movie's title and overview. It is a **multi-label text
classification** system: every movie may have several correct genres, and each
of the 19 genres is modeled as an independent binary decision.

This document describes the complete path from raw source records to the saved
model and archived run metadata. The executable training entry point is
`train.py`; data preparation is performed separately by `data.py`.

The current source code is the final authority if this document and the code
ever differ. This version documents `MAX_LENGTH = 256`.

## 2. End-to-End Flow

The complete lifecycle is:

```text
data/moviedb.movies.csv + data/genres.csv
                    |
                    | python data.py
                    v
          data/cleaned_movies.csv
                    |
                    | python train.py
                    v
  load and filter predefined train/validation/test splits
                    |
                    v
  tokenize plot text + create 19-value multi-hot labels
                    |
                    v
  calculate positive-class weights from original training split
                    |
                    v
  add minority-biased duplicate training examples
                    |
                    v
  fine-tune BERT with weighted BCE or weighted focal loss
                    |
                    v
  evaluate and checkpoint every 2,500 optimizer steps
                    |
                    v
  restore the checkpoint with the best validation macro F1
                    |
                    v
  tune one decision threshold per genre on validation data
                    |
                    v
  evaluate once on held-out test data using validation thresholds
                    |
                    v
  save model, tokenizer, metrics, plots, and run history
```

The test split does not select the model checkpoint or the thresholds used in
the final reported test metrics.

## 3. Software and Runtime Requirements

The project metadata requires Python 3.12 or newer. The training stack includes:

| Component | Declared version or constraint | Role |
|---|---:|---|
| Python | `>=3.12` | Runtime |
| PyTorch | `2.11.0` | Tensor operations, GPU execution, and loss computation |
| Transformers | `>=5.5.0` with `torch` extra | BERT, Trainer, and training arguments |
| Datasets | `>=4.0.0` | CSV loading, filtering, mapping, and dataset selection |
| NumPy | `>=2.0.0` | Metrics, threshold search, and weighted sampling |
| Accelerate | `>=1.14.0` | Trainer device and distributed runtime support |
| Matplotlib | `>=3.11.1` | Training and evaluation plots |
| Polars | `>=1.44.1` | Upstream data cleaning and validation |

On Windows, PyTorch is configured through the `pytorch-cu130` package index.
Training has `fp16=True`, so the intended environment is a CUDA-capable NVIDIA
GPU with a compatible driver. CPU-only execution or unsupported hardware can
fail because FP16 is enabled unconditionally.

Install the locked project environment and run training with:

```powershell
uv sync
uv run python data.py
uv run python validation.py
uv run python train.py
```

`data.py` and `validation.py` are preparation checks; `train.py` does not call
them automatically.

## 4. Configuration Reference

### 4.1 Model, paths, and labels

| Setting | Value | Meaning |
|---|---|---|
| `MODEL_NAME` | `google-bert/bert-base-uncased` | Pretrained tokenizer and encoder checkpoint |
| `DATASET_PATH` | `data/cleaned_movies.csv` | Model-ready CSV input |
| `OUTPUT_DIR` | `outputs/bert-movie-genres` | Shared checkpoints and latest successful model |
| `RUNS_DIR` | `outputs/bert-movie-genres/runs` | Per-attempt metrics and plots |
| `HISTORY_PATH` | `outputs/bert-movie-genres/training_history.jsonl` | Append-only run summary history |
| `MAX_LENGTH` | `256` | Maximum number of BERT tokens per example |
| Number of labels | `19` | Width of labels and classifier output |

The label order is part of the model contract:

| ID | Genre | ID | Genre |
|---:|---|---:|---|
| 0 | Adventure | 10 | Music |
| 1 | War | 11 | Western |
| 2 | Documentary | 12 | Animation |
| 3 | Science Fiction | 13 | Mystery |
| 4 | Thriller | 14 | TV Movie |
| 5 | Family | 15 | Horror |
| 6 | Romance | 16 | Comedy |
| 7 | Fantasy | 17 | Crime |
| 8 | Drama | 18 | History |
| 9 | Action |  |  |

`LABEL_TO_ID` maps these names to indices, and `ID_TO_LABEL` reverses the
mapping. Both mappings are written into the Hugging Face model configuration.
Changing the order without rebuilding the dataset/model contract would assign
the wrong semantic meaning to classifier outputs.

### 4.2 Imbalance and threshold settings

| Setting | Value | Meaning |
|---|---:|---|
| `LOSS_TYPE` | `weighted_bce` | Active objective; `focal` is the alternative |
| `FOCAL_GAMMA` | `2.0` | Focal modulation exponent when focal loss is selected |
| `MAX_POS_WEIGHT` | `10.0` | Maximum positive-class loss weight |
| `OVERSAMPLE_RATIO` | `0.5` | Number of added samples as a fraction of original train size |
| `OVERSAMPLE_POWER` | `0.5` | Strength of inverse-prevalence sampling bias |
| `RANDOM_SEED` | `42` | NumPy oversampling seed |
| `THRESHOLD_CANDIDATES` | `0.10` through `0.90`, step `0.05` | Per-label F1 threshold grid |
| `LABEL_THRESHOLD` | `0.5` | Reference line in the threshold plot |

`LABEL_THRESHOLD` is not the final decision rule and is not used to choose the
best checkpoint. The code tunes thresholds whenever `compute_metrics` is called
without an explicit threshold vector. Its direct use is the red "Default" line
in `threshold_tuning.png`.

### 4.3 Trainer settings

| Argument | Value | Effect |
|---|---:|---|
| `num_train_epochs` | `3` | Maximum passes over the oversampled training dataset |
| `learning_rate` | `2e-5` | Peak fine-tuning learning rate |
| `per_device_train_batch_size` | `16` | Examples per training device and forward pass |
| `per_device_eval_batch_size` | `32` | Examples per evaluation device and forward pass |
| `weight_decay` | `0.01` | AdamW-style parameter regularization |
| `warmup_ratio` | `0.1` | Fraction of training steps used for LR warmup |
| `eval_strategy` | `steps` | Evaluate by optimizer-step interval |
| `eval_steps` | `2500` | Validation interval |
| `save_strategy` | `steps` | Save by optimizer-step interval |
| `save_steps` | `2500` | Checkpoint interval, aligned with evaluation |
| `logging_steps` | `100` | Trainer logging interval |
| `load_best_model_at_end` | `True` | Restore best saved checkpoint after training |
| `metric_for_best_model` | `macro_f1` | Validation metric used to rank checkpoints |
| `greater_is_better` | `True` | Larger macro F1 wins |
| `fp16` | `True` | Use FP16 mixed precision |
| `save_total_limit` | `2` | Retain at most two Trainer checkpoints, subject to best-checkpoint retention |
| `report_to` | `none` | Disable external experiment trackers |
| Early-stopping patience | `3` evaluations | Stop after three validation events without improvement |

No gradient accumulation value is set, so the effective batch size is normally:

$$
B_{effective}=16\times N_{devices}
$$

for ordinary data-parallel training. Distributed runtime details may alter the
exact global batching behavior.

## 5. Data Preparation Before Training

### 5.1 Raw inputs

`data.py` reads:

- `data/moviedb.movies.csv`, containing movie metadata.
- `data/genres.csv`, defining the accepted genre-name-to-ID mapping.

It writes `data/cleaned_movies.csv` with exactly these columns:

```text
plot,genre_ids,genre_names,group_id,split
```

Training reads `plot`, `genre_names`, and `split`. It removes all original
columns after encoding, so `genre_ids` and `group_id` are not consumed directly
by the model pipeline.

### 5.2 Text normalization and rejection

The cleaner casts title, original title, overview, and genre names to strings,
collapses repeated whitespace, and trims surrounding whitespace. Empty and
placeholder-like values such as `null`, `none`, `nan`, and `n/a` become missing.

The title falls back to `original_title` if `title` is unavailable. A row is
rejected if title, overview, or genres remain missing. It is also rejected when:

- The lowercase overview exactly matches a known placeholder.
- The overview starts with `no overview` or `.....`.
- The overview contains fewer than 50 characters.
- The overview contains fewer than 8 non-whitespace words.

Genre strings are split on commas and checked against `genres.csv`. Any unknown
genre stops preprocessing with a `ValueError`.

### 5.3 Deduplication and label consolidation

Rows are grouped by lowercase title and lowercase overview. Each group keeps the
first normalized title and overview and merges its unique genre names. Groups
with zero genres or more than six genres are removed.

The final model text is constructed exactly as:

```text
Title: <normalized title> Overview: <normalized overview>
```

This explicit prefixing gives the encoder a stable boundary between title and
overview, although BERT receives the result as one sequence rather than as a
pair of separately segmented input strings.

### 5.4 Stable split assignment

The cleaner lowercases the overview and hashes it with BLAKE2b using an 8-byte
digest. That unsigned 64-bit value is `group_id`. The split is selected by the
last decimal bucket:

```text
group_id % 10 == 0  -> test
group_id % 10 == 1  -> validation
group_id % 10 in 2..9 -> train
```

The expected distribution is approximately 80% train, 10% validation, and 10%
test. Exact counts depend on the hash outcomes. Because identical normalized
overviews receive the same `group_id`, overview duplicates cannot be assigned
to different splits. The split is deterministic across runs and does not rely
on the training random seed.

`validation.py` checks schema, unique plots, group isolation, non-null splits,
placeholder removal, maximum genres, complete label coverage, and per-split
genre support.

## 6. Run Initialization and Status Tracking

Execution begins in `main()`. It captures the current UTC time and formats a run
ID with microsecond precision:

```text
YYYYMMDDTHHMMSS.ffffffZ
```

For example:

```text
20260906T141530.123456Z
```

It creates:

```text
outputs/bert-movie-genres/runs/<run-id>/
```

The initial `run_summary.json` contains `status: "running"`, the run ID, start
timestamp, and base model name. Creating the run directory uses no
`exist_ok=True`; an extremely unlikely timestamp collision fails before the
training exception handler begins.

`train_run()` performs all remaining successful-run work. `main()` wraps that
call in `try/except BaseException` so ordinary exceptions and keyboard
interrupts are archived before being re-raised.

## 7. Tokenizer and Model Initialization

The tokenizer is loaded with:

```python
AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

The model is loaded with a sequence-classification head configured for 19
outputs:

```python
AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=19,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
    problem_type="multi_label_classification",
)
```

BERT Base has a hidden width of 768. Conceptually, the forward path is:

```text
token IDs and attention information: [B, L]
                         |
                         v
              BERT encoder representation
                         |
                         v
            sequence classification head
                         |
                         v
                    logits: [B, 19]
```

Here $B$ is batch size and $L\leq256$ is the dynamically padded sequence length
for the current batch. The tokenizer call itself does not request fixed-length
padding. Trainer's collator pads examples to the longest sequence in each batch,
subject to the 256-token truncation limit.

The classification head is newly initialized because the base checkpoint is not
already trained for this project's 19-label task. BERT's encoder starts with
pretrained language representations; all model parameters are fine-tuned unless
Transformers applies a model-specific default not overridden here. The script
does not freeze layers.

## 8. Dataset Loading, Splitting, and Encoding

### 8.1 Loading and filtering

Hugging Face Datasets loads the entire CSV as one in-memory logical dataset:

```python
load_dataset("csv", data_files=DATASET_PATH, split="train")
```

The `split="train"` argument here means "return the loaded CSV dataset"; it does
not create a new random train split. The code then filters the CSV's existing
`split` column into `train`, `validation`, and `test` datasets. Each filter uses
two worker processes.

### 8.2 Text tokenization

`encode_batch()` receives batches of examples and tokenizes `examples["plot"]`
with:

```python
tokenizer(plot_texts, max_length=256, truncation=True)
```

For BERT, the resulting example normally includes:

- `input_ids`: vocabulary IDs including special tokens.
- `attention_mask`: 1 for real tokens and 0 for batch padding.
- `token_type_ids`: segment IDs when supplied by the tokenizer.

Inputs longer than 256 tokens are cut off. Since the title comes first, it is
normally preserved while the tail of a long overview is removed. Inputs shorter
than 256 are not expanded during the map operation; padding happens per batch.

Uncased BERT normalizes case through its tokenizer. `MAX_LENGTH` counts subword
tokens and special tokens, not characters, words, or bytes.

### 8.3 Multi-hot label encoding

For each comma-separated `genre_names` value, `encode_batch()` creates a
19-element floating-point vector initialized to zero. Every listed genre sets
its mapped position to `1.0`.

For an Action and Comedy movie, only IDs 9 and 16 are active:

```text
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

This is multi-hot rather than one-hot encoding. Multiple ones are valid. An
unknown or misspelled genre raises `ValueError` instead of being ignored.

After mapping, the original CSV columns are removed. The model-facing dataset
contains tokenized inputs and `labels`. Conceptual batch shapes are:

| Tensor | Shape | Typical dtype | Meaning |
|---|---|---|---|
| `input_ids` | $[B,L]$ | integer | BERT vocabulary IDs |
| `attention_mask` | $[B,L]$ | integer | Real-token/padding mask |
| `token_type_ids` | $[B,L]$ | integer | BERT segment IDs, when emitted |
| `labels` | $[B,19]$ | floating point | Multi-hot genre targets |
| output `logits` | $[B,19]$ | model floating point | Unbounded genre scores |

The map operation is batched, removes the original columns using the training
split's schema, and uses two worker processes.

## 9. Class-Imbalance Handling

The pipeline combines two different techniques: positive-class loss weighting
and train-only oversampling. They are deliberately computed/applied before the
Trainer starts.

### 9.1 Positive-class weights

Weights are calculated from the **original encoded training split before
oversampling**. For genre $c$ with $P_c$ positive examples among $N$ original
training examples:

$$
N_c^- = N-P_c
$$

$$
w_c^+ = \operatorname{clip}\left(\frac{N_c^-}{P_c},1,10\right)
$$

The lower bound of 1 prevents common genres from receiving a positive weight
below the ordinary BCE value. The upper bound prevents extremely rare genres
from receiving an arbitrarily large multiplier. If any genre has no positive
training example, training stops with a message listing the missing genres.

PyTorch's `pos_weight` scales only the positive term of binary cross entropy; it
does not multiply negative examples by the same value.

### 9.2 Minority-biased oversampling

Oversampling modifies only the training split. Validation and test distributions
remain untouched.

For each genre $c$, prevalence is:

$$
q_c=\frac{P_c}{N}
$$

The sampling label weight is:

$$
a_c=\max\left(q_c,\frac{1}{N}\right)^{-0.5}
$$

For example $s$ with labels $y_{s,c}\in\{0,1\}$, the sample weight is the
largest weight among its active genres:

$$
a_s=\max_c(y_{s,c}a_c)
$$

This maximum means one rare label can make a multi-label movie likely to be
sampled; the weights are not summed across all of its labels. Sampling
probabilities are normalized:

$$
\Pr(s)=\frac{a_s}{\sum_j a_j}
$$

The number of additional examples is:

$$
N_{extra}=\operatorname{round}(0.5N)
$$

Indices are drawn with replacement using NumPy's random generator seeded with
42. The original $N$ indices and sampled indices are concatenated and shuffled.
The resulting training set is approximately $1.5N$ examples. An "epoch" in the
Trainer therefore means one pass over this enlarged dataset, not the original
unique set.

Setting `OVERSAMPLE_RATIO <= 0` disables this stage. Positive weighting remains
active unless the Trainer implementation is changed.

### 9.3 Combined effect

Rare labels influence training twice:

1. Examples containing rare genres are more likely to be duplicated.
2. Positive errors for rare genres receive larger loss weights.

This can improve minority recall and macro F1, but it can also overemphasize
rare labels, increase false positives, or destabilize optimization. The cap of
10 limits only the loss-weighting side, not the combined effect. Per-label
precision, recall, F1, and average precision should therefore be inspected after
every run.

## 10. Loss Computation

`ImbalanceAwareTrainer` subclasses Hugging Face `Trainer` and overrides
`compute_loss()`. It accepts only `weighted_bce` and `focal` as loss types.

The method removes `labels` from the model input dictionary, performs the model
forward pass, and obtains logits $x_{b,c}$ of shape $[B,19]$. Labels are moved to
the logits' device and dtype.

### 10.1 Weighted binary cross entropy

Let $y_{b,c}\in\{0,1\}$, $p_{b,c}=\sigma(x_{b,c})$, and $w_c^+$ be the positive
weight. The unreduced element loss is:

$$
\ell_{b,c}=-\left[w_c^+y_{b,c}\log(p_{b,c})+
(1-y_{b,c})\log(1-p_{b,c})\right]
$$

PyTorch computes this with `binary_cross_entropy_with_logits`, which combines
sigmoid and BCE in a numerically stable operation. No softmax is used: softmax
would force labels to compete for a fixed probability mass and would be wrong
for movies with multiple genres.

The final scalar loss is the arithmetic mean across every batch item and every
genre:

$$
\mathcal{L}=\frac{1}{19B}\sum_{b=1}^{B}\sum_{c=1}^{19}\ell_{b,c}
$$

### 10.2 Optional focal modulation

When `LOSS_TYPE = "focal"`, weighted BCE is still the base loss. The code defines
the probability assigned to the correct binary outcome as:

$$
p_{t,b,c}=\begin{cases}
p_{b,c} & y_{b,c}=1\\
1-p_{b,c} & y_{b,c}=0
\end{cases}
$$

It then applies:

$$
\ell_{b,c}^{focal}=\ell_{b,c}(1-p_{t,b,c})^{2}
$$

Easy, confidently correct decisions receive a small multiplier; hard or
incorrect decisions remain influential. The implementation does not add a
separate focal alpha because `pos_weight` already supplies class-sensitive
positive weighting.

The currently active objective is `weighted_bce`, so `FOCAL_GAMMA` has no effect
unless `LOSS_TYPE` is changed.

## 11. Optimization and Trainer Lifecycle

`Trainer` creates the optimizer and scheduler from `TrainingArguments`. With the
current configuration, fine-tuning targets a peak learning rate of $2\times
10^{-5}$, warms up over the first 10% of estimated optimizer steps, and applies
0.01 weight decay to eligible parameters.

At a high level, one optimizer step is:

```text
collate and dynamically pad 16 examples per device
    -> BERT forward pass under FP16 mixed precision
    -> custom weighted loss
    -> backward pass
    -> mixed-precision gradient handling
    -> optimizer update
    -> learning-rate scheduler update
    -> clear/replace gradients for next step
```

The code does not explicitly configure gradient accumulation, gradient
checkpointing, custom optimizer parameters, freezing, or resume-from-checkpoint.
A new invocation calls `trainer.train()` without a checkpoint argument and
starts a fresh fine-tune from the base model.

Every 100 optimizer steps, Trainer records logs. Every 2,500 steps, it evaluates
on validation data and saves a checkpoint to `OUTPUT_DIR`. Because evaluation
and saving intervals match, each candidate metric has a corresponding saved
checkpoint.

## 12. Evaluation During Training

### 12.1 Logits to probabilities

For every genre, logits are converted independently with sigmoid:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The resulting probabilities have shape $[N_{eval},19]$.

### 12.2 Per-label threshold search at every evaluation

When Trainer calls `compute_metrics`, no threshold vector is supplied. Contrary
to a fixed-0.5 evaluation, the function calls `find_per_label_thresholds()`.
For each genre independently, it evaluates these 17 candidates:

```text
0.10, 0.15, 0.20, ..., 0.85, 0.90
```

For a candidate $t_c$, prediction is positive when $p_{n,c}\geq t_c$. The
candidate yielding the largest validation F1 for that genre is selected. If
multiple candidates tie, `numpy.argmax` selects the first, which is the lowest
threshold among the tied candidates.

These temporary per-label thresholds are recomputed for every validation event.
The returned `macro_f1` is therefore the macro F1 achievable on that validation
snapshot under its independently optimized grid thresholds.

### 12.3 Confusion counts and F1

For genre $c$:

$$
TP_c=\sum_n[\hat y_{n,c}=1\land y_{n,c}=1]
$$

$$
FP_c=\sum_n[\hat y_{n,c}=1\land y_{n,c}=0]
$$

$$
FN_c=\sum_n[\hat y_{n,c}=0\land y_{n,c}=1]
$$

Precision and recall are:

$$
Precision_c=\frac{TP_c}{TP_c+FP_c}
$$

$$
Recall_c=\frac{TP_c}{TP_c+FN_c}
$$

Undefined divisions are reported as zero. Per-label F1 is:

$$
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}
$$

Macro F1 gives each genre equal weight:

$$
MacroF1=\frac{1}{19}\sum_{c=1}^{19}F1_c
$$

Micro F1 pools decisions before computing the ratio:

$$
MicroF1=\frac{2\sum_cTP_c}
{2\sum_cTP_c+\sum_cFP_c+\sum_cFN_c}
$$

Macro F1 is sensitive to rare-genre quality; micro F1 is dominated more strongly
by common labels and the total number of correct decisions.

### 12.4 Average precision

Average precision is computed independently for each genre without thresholding.
Examples are sorted by descending predicted probability. At each rank containing
a true positive, precision at that rank is recorded; the mean over all actual
positives is returned:

$$
AP_c=\frac{1}{P_c}\sum_{k:y_{(k),c}=1}
\frac{\sum_{j=1}^{k}y_{(j),c}}{k}
$$

If an evaluated split has no positive example for a genre, its average precision
is defined as zero. This implementation does not calculate a single mean average
precision field; it emits one AP value per label.

### 12.5 Best-checkpoint selection and early stopping

Trainer prefixes returned fields with `eval_`, so `macro_f1` becomes
`eval_macro_f1`. The checkpoint with the highest threshold-optimized validation
macro F1 is considered best. `load_best_model_at_end=True` restores that
checkpoint after training ends.

Early stopping allows three consecutive evaluation events without improvement.
At 2,500 steps per event, its nominal patience is 7,500 optimizer steps, but it
is fundamentally a count of evaluation events. If fewer than four total
evaluations occur, early stopping may never have enough events to trigger. A
normal early stop is recorded as a completed run, not a failure.

## 13. Final Threshold Tuning

After training, the in-memory model is the restored best checkpoint. The script
runs `trainer.predict()` on the validation split. It then explicitly repeats the
same per-label grid search in `tune_thresholds()` and saves the selected vector.

`thresholds.json` contains:

```json
{
  "per_label_thresholds": {
    "Adventure": 0.4,
    "War": 0.25
  },
  "validation_micro_f1": 0.0,
  "validation_macro_f1": 0.0
}
```

The snippet is structural; the numbers are examples, not published model
results. Actual output includes all 19 labels and measured validation scores.

Threshold optimization maximizes each genre's validation F1 separately. It does
not jointly optimize micro F1, calibrate probabilities, or enforce a minimum or
maximum number of genres per movie. Thresholds are decision boundaries, not a
guarantee that sigmoid values are calibrated probabilities.

`threshold_tuning.png` plots the 19 selected thresholds as horizontal bars and a
dashed reference line at 0.5.

## 14. Held-Out Test Evaluation

The restored best model predicts the untouched test split. Final test metrics are
calculated by calling `compute_metrics` with the **validation-derived threshold
vector supplied explicitly**. Because thresholds are supplied, no test-based
threshold search occurs in this final metric calculation.

The reported fields are:

- Overall `micro_f1` and `macro_f1`.
- Precision, recall, F1, and average precision for each of 19 genres.

That produces 78 values: 2 aggregate metrics plus $19\times4$ per-label metrics.
They are written to the run's `test_metrics.json` and included in its summary.

There is one implementation subtlety: `trainer.predict(dataset["test"])` also
invokes the registered `compute_metrics` callback internally with no explicit
threshold vector. It consequently computes a test-optimized metric dictionary
inside `test_predictions.metrics`. The pipeline discards that dictionary and
recomputes the saved metrics with validation thresholds, so it does not affect
checkpoint selection, thresholds, or reported test scores. It is nevertheless
unnecessary work and means test labels are inspected by a transient metric
calculation before the proper final calculation.

## 15. Plots and Diagnostic Outputs

### 15.1 Training overview

`training_metrics.png` is a 2-by-2 figure containing:

- Training loss.
- Validation loss.
- Validation micro F1.
- Validation macro F1.

The horizontal value is `epoch` when Trainer supplied it, otherwise `step`.
Panels with no matching records display "No data collected".

### 15.2 Final per-label metrics

`per_label_metrics.png` displays grouped horizontal bars for precision, recall,
F1, and average precision for every genre. Its title refers to the best
checkpoint because that model is restored before final prediction.

### 15.3 Per-label history

`per_label_metric_history.png` contains two heatmaps over validation events:

- Per-genre F1 over training steps.
- Per-genre average precision over training steps.

Both use a fixed 0-to-1 color scale. At most roughly ten x-axis positions are
labeled to keep long histories readable. If no evaluation record contains
`eval_macro_f1`, the plot is skipped.

### 15.4 Threshold plot

`threshold_tuning.png` compares final validation-selected thresholds across
genres with the 0.5 reference line.

## 16. Saving, Checkpoints, and Run Archives

### 16.1 Shared Trainer checkpoints

During training, Hugging Face Trainer writes `checkpoint-<step>` directories
directly under:

```text
outputs/bert-movie-genres/
```

They are not stored inside the run-specific archive. A checkpoint may include
model weights, optimizer state, scheduler state, gradient-scaler state, random
state, tokenizer files, and Trainer state. `save_total_limit=2` controls retained
checkpoints. Because every run shares `OUTPUT_DIR`, sequential runs may replace
or prune checkpoints from earlier runs.

The script has no concurrency lock itself. Concurrent local invocations would
write to the same checkpoint and latest-model paths and are unsafe. The release
workflow supplies its own concurrency control.

### 16.2 Per-run archive

On a successful run, its directory contains small reproducibility and analysis
artifacts:

```text
runs/<run-id>/
|-- run_summary.json
|-- trainer_log_history.json
|-- thresholds.json
|-- test_metrics.json
|-- training_metrics.png
|-- per_label_metrics.png
|-- per_label_metric_history.png   # when evaluation history exists
`-- threshold_tuning.png
```

Model weights are intentionally not duplicated into every run directory.

### 16.3 Latest successful model

After metrics and plots are generated, the restored model and tokenizer are
saved to the shared output root. The script then copies these run artifacts to
that same root when they exist:

```text
thresholds.json
test_metrics.json
training_metrics.png
per_label_metrics.png
per_label_metric_history.png
threshold_tuning.png
```

Therefore, the root model files and analysis artifacts represent the latest
successfully completed save sequence. A failure before this stage normally
leaves the preceding successful model and artifacts in place.

### 16.4 Completed summary

The completed `run_summary.json` records:

- Run ID, status, UTC start time, and UTC completion time.
- Base model name.
- Custom constants such as max length, loss mode, imbalance controls, and seed.
- Full serialized `TrainingArguments`.
- Dataset sizes before and after oversampling and validation/test sizes.
- Trainer's training metrics.
- Validation prediction metrics.
- Final validation threshold metrics.
- Correct validation-threshold-based test metrics.
- Best checkpoint path and best metric.

`archive_run_summary()` writes that JSON, appends the same summary as one compact
line in `training_history.jsonl`, and copies it to `latest_run.json`.

Because `latest_run.json` is updated for every archived outcome, it means
"latest attempted run," not necessarily "latest successful model."

## 17. Failure and Interruption Behavior

Any exception raised inside `train_run()` is caught by `main()`:

- `KeyboardInterrupt` becomes status `interrupted`.
- Every other `BaseException` becomes status `failed`.

The archived failure summary contains run ID, status, timestamps, model name,
exception class name, and exception message. It replaces the initial `running`
summary, is appended to `training_history.jsonl`, and becomes `latest_run.json`.
The exception is then re-raised, so the command still exits unsuccessfully.

Failure summaries do not include all configuration, dataset sizes, partial
metrics, or the best checkpoint. Files already emitted by Trainer may remain in
the shared output root. There is no automatic cleanup or resume logic.

Potential failures include missing CSV/model files, unknown genres, a genre with
zero training positives, CUDA or FP16 incompatibility, GPU memory exhaustion,
disk exhaustion, and plotting or serialization errors.

## 18. Reproducibility Boundaries

The data split is deterministic because it is hash-based. Oversampling is also
deterministic for a fixed encoded training dataset because it uses
`np.random.default_rng(42)`.

The script does **not** pass `seed=RANDOM_SEED` to `TrainingArguments`. Trainer
therefore uses its own default seed (commonly 42 for the documented Transformers
version), but that value is not linked programmatically to `RANDOM_SEED`. The
script also does not explicitly enable deterministic PyTorch algorithms.

Exact reruns can still vary due to:

- GPU kernel nondeterminism.
- Library, CUDA, driver, or hardware changes.
- Base-model or tokenizer revision changes because no immutable revision is
  pinned.
- Dataset content changes at the same path.
- Multiprocessing and distributed-runtime differences.
- FP16 numerical behavior.

For stronger reproducibility, record the environment lock, source commit,
dataset checksum, base-model revision, driver/GPU details, and all random seeds.
Those items are not all stored in the current run summary.

## 19. Computational Implications of `MAX_LENGTH = 256`

Self-attention has approximately quadratic sequence-length cost:

$$
O(L^2)
$$

Reducing maximum length from 384 to 256 changes the attention-matrix area by:

$$
\frac{256^2}{384^2}=\frac{4}{9}\approx0.444
$$

For examples that previously reached 384 tokens, attention-score memory and
work are therefore roughly 44.4% of the former 384-token area, a reduction of
about 55.6%. End-to-end speed and memory do not improve by exactly that amount
because embeddings, feed-forward layers, padding patterns, data loading, and
other operations have different scaling.

The tradeoff is information loss for plots exceeding 256 subword tokens. Since
truncation removes the tail, late plot details may be unavailable to the model.

## 20. Interpretation and Leakage Boundaries

The pipeline observes each split as follows:

| Split | Gradient updates | Checkpoint selection | Threshold selection | Final reporting |
|---|---:|---:|---:|---:|
| Train | Yes | No | No | Train metrics only |
| Validation | No | Yes | Yes | Validation metrics |
| Test | No | No | No for saved results | Final test metrics |

Threshold tuning on validation is valid model selection, but repeatedly tuning
19 thresholds can overfit a small validation set. Test metrics are the main
estimate of generalization. Repeated human decisions based on test results would
eventually turn the test set into another validation set; a new final holdout
would then be required.

Hash grouping limits leakage from identical normalized overviews. It does not
detect paraphrases, remakes with similar descriptions, franchise overlap, or
other semantic near-duplicates.

## 21. Known Design Caveats

1. Positive weighting and oversampling can compound minority emphasis.
2. Best-checkpoint macro F1 is threshold-optimized separately at every
   validation event, not measured under one fixed deployment threshold vector.
3. Final thresholds are selected after checkpoint selection; jointly selecting
   a checkpoint and stable threshold vector could choose a different result.
4. The transient metrics generated by `trainer.predict(test)` optimize on test
   labels but are discarded; only recomputed validation-threshold metrics are
   saved.
5. `fp16=True` is unconditional and limits hardware portability.
6. No base-model revision or cleaned-dataset checksum is recorded.
7. No explicit training resume behavior is implemented.
8. Shared checkpoint paths make concurrent runs unsafe and do not preserve each
   run's checkpoint weights.
9. Average precision is implemented locally rather than through a standard
   metrics library; behavior for tied scores follows NumPy's sort ordering.
10. Threshold candidates are restricted to 0.10 through 0.90, so an optimum
    outside that grid cannot be selected.
11. `training_history.jsonl` is append-only and has no file lock.
12. The model predicts labels independently and does not explicitly enforce
    valid genre combinations.

## 22. Comparing and Releasing Runs

After at least two completed runs, compare the latest pair with:

```powershell
uv run python comparison/compare_runs.py
```

The comparison uses held-out test macro F1 as its primary verdict and test micro
F1 as a tie-breaker. Failed and interrupted runs remain in history but are
excluded from comparisons.

The automated release workflow restores prior history, trains on a self-hosted
Windows GPU runner, compares runs, uploads the latest model to Hugging Face,
tags the model revision as `model-<run-id>`, and creates a matching GitHub
Release. Publishing occurs only after successful training. Hugging Face model
weights are kept there rather than attached to GitHub Releases.

## 23. Practical Run Review Checklist

Before training:

- Confirm `data/cleaned_movies.csv` was regenerated from the intended source.
- Run `validation.py` and inspect split and label-support output.
- Confirm CUDA and FP16 support.
- Confirm enough disk space for two checkpoints plus the final model.
- Record the source commit and dependency lock state.

After training:

- Confirm `latest_run.json` has status `completed`.
- Inspect best checkpoint and best validation macro F1.
- Compare train and validation loss for instability or overfitting.
- Compare macro F1 with micro F1; a large gap often signals rare-label weakness.
- Inspect every genre's precision, recall, F1, and average precision.
- Check whether thresholds sit at grid boundaries 0.10 or 0.90.
- Review threshold and per-label history plots for unstable labels.
- Compare against the previous completed run before release.
- Treat the saved test metrics as final evaluation, not another tuning target.

## 24. Function-by-Function Index

| Function or class | Responsibility |
|---|---|
| `json_default` | Convert NumPy scalars/arrays and paths for JSON serialization |
| `write_json` | Write indented UTF-8 JSON using the custom serializer |
| `archive_run_summary` | Save one summary, append history, and update latest attempt |
| `label_metric_name` | Convert genre names into metric-key-safe lowercase names |
| `multilabel_f1` | Compute confusion counts, per-label F1, and micro F1 |
| `average_precision` | Compute ranking-based AP for one genre |
| `encode_batch` | Tokenize text and create 19-value multi-hot targets |
| `find_per_label_thresholds` | Grid-search the best F1 threshold per genre |
| `compute_metrics` | Produce aggregate and per-label evaluation metrics |
| `calculate_pos_weights` | Derive capped positive BCE weights from original train data |
| `oversample_minority_examples` | Add reproducible rare-label-biased train examples |
| `ImbalanceAwareTrainer` | Replace Trainer's loss with weighted BCE/focal loss |
| `tune_thresholds` | Save final validation thresholds and threshold plot |
| `plot_training_metrics` | Plot loss and aggregate validation history |
| `plot_per_label_metrics` | Plot final held-out metrics by genre |
| `plot_per_label_history` | Plot validation F1/AP heatmaps over checkpoints |
| `train_run` | Execute load, train, evaluate, save, and successful archival |
| `main` | Create run identity and archive failed/interrupted outcomes |

## 25. Summary

Version 0.1.0 trains an unfrozen BERT Base encoder and 19-output classification
head on 256-token movie title/overview sequences. It addresses imbalance through
both capped positive BCE weights and deterministic minority oversampling. Model
selection uses validation macro F1 with per-label grid-searched thresholds,
final thresholds are persisted from the restored best checkpoint's validation
predictions, and held-out test results are saved using those validation-derived
thresholds. Every attempt receives a durable status record; successful runs add
metrics and plots while the shared output root holds the latest successful model
and Trainer checkpoints.