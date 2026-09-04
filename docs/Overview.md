Updated todo list

## Overview

`train.py` fine-tunes BERT to predict **multiple genres** from each movie’s combined title and overview. Because one movie can belong to several genres, this is **multi-label classification**, not ordinary single-label classification.

## Imports

- `from pathlib import Path`  
  Imports `Path`, an object-oriented way to construct filesystem paths. It handles Windows and Unix path separators safely.

- `import numpy as np`  
  Imports NumPy as `np`. It is used for vectorized sigmoid calculations, Boolean comparisons, and F1 metrics.

- `from datasets import load_dataset`  
  Imports Hugging Face’s dataset loader. It reads the CSV and provides splitting and parallel preprocessing.

- `from transformers import (...)`  
  Imports the Hugging Face Transformers components:
  - `AutoTokenizer`: converts text into BERT token IDs.
  - `AutoModelForSequenceClassification`: loads BERT with a classification head.
  - `Trainer`: manages training and evaluation.
  - `TrainingArguments`: stores training configuration.

## Constants

- `MODEL_NAME = "google-bert/bert-base-uncased"`  
  Selects pretrained BERT Base Uncased. “Uncased” means text capitalization is ignored. Pretraining gives the model existing language knowledge.

- `DATASET_PATH = Path(__file__).with_name("cleaned_movies.csv")`  
  Finds the CSV beside `train.py`. `__file__` represents the current script’s path, so training does not depend on the terminal’s working directory.

- `OUTPUT_DIR = Path(__file__).with_name("outputs") / "bert-movie-genres"`  
  Constructs the directory where checkpoints and the final model are saved. `/` joins path components when using `Path`.

- `MAX_LENGTH = 256`  
  Limits each input to 256 tokens. This reduces memory and computation. Longer inputs are truncated.

- `LABEL_THRESHOLD = 0.5`  
  A genre is predicted when its probability is at least 50%.

## Genre Labels

- `CLASS_LABELS = [...]`  
  Defines the 19 possible output genres. The order is important because each position corresponds to one output neuron.

- `LABEL_TO_ID = {label: index for index, label in enumerate(CLASS_LABELS)}`  
  Creates a mapping such as:

  ```python
  {"Adventure": 0, "War": 1, ..., "History": 18}
  ```

  `enumerate()` produces each label together with its numeric position.

- `ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}`  
  Reverses that mapping:

  ```python
  {0: "Adventure", 1: "War", ..., 18: "History"}
  ```

  These mappings are stored in the model configuration, making predictions easier to interpret.

## `encode_batch`

- `def encode_batch(examples, tokenizer):`  
  Defines preprocessing for a batch of dataset rows.

- `encoded = tokenizer(...)`  
  Converts each `title_overview` string into model inputs such as `input_ids` and `attention_mask`.

- `max_length=MAX_LENGTH`  
  Sets the maximum sequence length to 256 tokens.

- `truncation=True`  
  Removes tokens beyond that limit. Without truncation, oversized examples could cause errors.

- `labels = []`  
  Creates a list to hold the encoded genre labels.

- `for genre_names in examples["genre_names"]:`  
  Processes the comma-separated genres from every row in the batch.

- `multi_hot = [0.0] * len(CLASS_LABELS)`  
  Creates a 19-element vector initially containing zeros.

  For example, a movie classified as Action and Comedy could become:

  ```text
  [0, 0, 0, ..., 1 for Action, ..., 1 for Comedy, ...]
  ```

  This is called **multi-hot encoding** because several positions can be active. One-hot encoding would allow only one active label.

- `for genre_name in genre_names.split(","):`  
  Splits a value such as `"Action, Comedy"` into separate genre strings.

- `genre_name = genre_name.strip()`  
  Removes surrounding whitespace so `" Comedy"` becomes `"Comedy"`.

- `if genre_name not in LABEL_TO_ID:`  
  Checks whether the dataset contains an unexpected genre.

- `raise ValueError(...)`  
  Stops training with a clear message instead of silently producing incorrect labels.

- `multi_hot[LABEL_TO_ID[genre_name]] = 1.0`  
  Marks the genre’s corresponding position as present.

- `labels.append(multi_hot)`  
  Adds the completed label vector to the batch.

- `encoded["labels"] = labels`  
  Adds targets to the tokenized inputs. `Trainer` recognizes the special key `labels` and passes it to the model for loss calculation.

- `return encoded`  
  Returns the model-ready batch.

## `compute_metrics`

- `def compute_metrics(eval_prediction):`  
  Defines the evaluation metrics that `Trainer` runs after each evaluation phase.

- `logits, labels = eval_prediction`  
  Separates raw model outputs from the correct targets.

  **Logits** are unrestricted scores, not probabilities. They may range from negative to positive infinity.

- `probabilities = 1.0 / (1.0 + np.exp(-logits))`  
  Applies the sigmoid function:

  $$\sigma(x)=\frac{1}{1+e^{-x}}$$

  Sigmoid converts every label’s logit independently into a probability from 0 to 1. Independent probabilities are essential because a movie may have several genres.

- `predictions = probabilities >= LABEL_THRESHOLD`  
  Converts probabilities into Boolean predictions using the 0.5 threshold.

- `targets = labels >= 0.5`  
  Converts floating-point target vectors into Boolean values.

- `true_positives = ...`  
  Counts cases where a genre was predicted and was actually present.

- `false_positives = ...`  
  Counts genres predicted by the model that were not present.

- `false_negatives = ...`  
  Counts genres that were present but missed by the model.

- `per_label_denominator = ...`  
  Calculates the denominator of the F1 formula for every genre:

  $$F_1=\frac{2TP}{2TP+FP+FN}$$

- `per_label_f1 = np.divide(...)`  
  Calculates F1 independently for each genre.

- `out=np.zeros_like(...)`  
  Uses zero as the default result where division cannot be performed.

- `where=per_label_denominator != 0`  
  Prevents division by zero for genres with no positive examples or predictions.

- `micro_denominator = (...)`  
  Combines counts from every genre before calculating F1.

- `micro_f1 = ...`  
  Calculates **micro F1**. Frequent genres have more influence because every prediction is counted together.

- `max(micro_denominator, 1)`  
  Prevents division by zero.

- `"micro_f1": float(micro_f1)`  
  Returns the combined F1 score.

- `"macro_f1": float(per_label_f1.mean())`  
  Returns the average of individual genre F1 scores. Every genre receives equal importance, including rare genres.

Using both is significant:

- **Micro F1** measures overall prediction quality.
- **Macro F1** reveals whether the model also performs well on uncommon genres.

## `main`

- `def main() -> None:`  
  Defines the main training workflow. `-> None` is a type annotation showing that the function does not return a value.

- `AutoTokenizer.from_pretrained(MODEL_NAME)`  
  Downloads or loads the tokenizer matching BERT. The tokenizer and model must use the same vocabulary.

- `AutoModelForSequenceClassification.from_pretrained(...)`  
  Loads pretrained BERT and adds a classification layer.

- `num_labels=len(CLASS_LABELS)`  
  Configures the classification layer to output 19 logits.

- `id2label=ID_TO_LABEL` and `label2id=LABEL_TO_ID`  
  Saves human-readable genre mappings in the model configuration.

- `problem_type="multi_label_classification"`  
  Tells Transformers to use multi-label behavior, normally including binary cross-entropy loss. Each genre is treated as an independent yes/no decision.

- `load_dataset("csv", ...)`  
  Loads the complete CSV as a Hugging Face `Dataset`.

- `split="train"`  
  Returns the loaded CSV directly as one dataset named conceptually as the training split.

- `dataset.train_test_split(test_size=0.05, seed=3407)`  
  Reserves 5% for evaluation and 95% for training. The fixed seed makes the split reproducible.

- `dataset.map(...)`  
  Applies `encode_batch` to both training and evaluation data.

- `batched=True`  
  Sends multiple rows to the preprocessing function at once, which is faster.

- `fn_kwargs={"tokenizer": tokenizer}`  
  Supplies the tokenizer as an additional function argument.

- `remove_columns=dataset["train"].column_names`  
  Removes original CSV columns after preprocessing, leaving only model inputs and labels.

- `num_proc=2`  
  Uses two worker processes for preprocessing.

## Trainer Configuration

- `Trainer(...)`  
  Creates Hugging Face’s high-level training controller.

- `model=model`  
  Specifies the model to train.

- `processing_class=tokenizer`  
  Gives the trainer the tokenizer, particularly for model saving and input processing support.

- `train_dataset=dataset["train"]`  
  Supplies the 95% training partition.

- `eval_dataset=dataset["test"]`  
  Supplies the 5% evaluation partition.

- `compute_metrics=compute_metrics`  
  Runs the custom micro and macro F1 calculations during evaluation.

- `output_dir=str(OUTPUT_DIR)`  
  Sets the checkpoint directory. Conversion to `str` ensures compatibility with the Transformers API.

- `num_train_epochs=3`  
  Passes over the complete training set three times.

- `learning_rate=2e-5`  
  Controls update size. A small learning rate is standard when fine-tuning pretrained BERT because large updates could damage its learned language representations.

- `per_device_train_batch_size=16`  
  Processes 16 training examples per GPU or CPU device before each update.

- `per_device_eval_batch_size=32`  
  Evaluates 32 examples at once. Evaluation needs less memory because gradients are not stored.

- `weight_decay=0.01`  
  Adds regularization to discourage excessively large weights and reduce overfitting.

- `warmup_ratio=0.1`  
  Gradually increases the learning rate during the first 10% of training. This stabilizes early fine-tuning.

- `eval_strategy="epoch"`  
  Evaluates the model after every epoch.

- `save_strategy="epoch"`  
  Saves a checkpoint after every epoch.

- `load_best_model_at_end=True`  
  Restores the best-performing checkpoint when training finishes instead of keeping only the final epoch.

- `metric_for_best_model="micro_f1"`  
  Uses evaluation micro F1 to decide which checkpoint is best.

- `greater_is_better=True`  
  Indicates that a larger micro F1 score is better.

- `fp16=True`  
  Uses 16-bit floating-point training. This generally speeds up supported NVIDIA GPUs and reduces GPU memory usage. It may fail on unsupported hardware.

- `logging_steps=100`  
  Reports training information every 100 update steps.

- `save_total_limit=2`  
  Keeps at most two checkpoints to limit disk usage. The best checkpoint is preserved.

- `report_to="none"`  
  Disables external experiment trackers such as Weights & Biases.

## Starting and Saving Training

- `trainer.train()`  
  Starts optimization, periodic evaluation, checkpointing, and best-model selection.

- `trainer.save_model(OUTPUT_DIR)`  
  Saves the final selected model and its configuration.

- `tokenizer.save_pretrained(OUTPUT_DIR)`  
  Saves the tokenizer alongside the model so inference uses exactly the same text processing.

- `if __name__ == "__main__":`  
  Runs training only when the file is executed directly. It does not start training if `train.py` is imported by another module. This guard is especially important with multiprocessing on Windows.

- `main()`  
  Calls the complete training workflow.