# Movie Genre Classification with BERT

This model predicts one or more movie genres from a movie's title and overview. It is a multi-label text-classification model built by fine-tuning `google-bert/bert-base-uncased` with Hugging Face Transformers.

For each input movie, the model produces an independent score for every genre. A movie can therefore be classified as both `Action` and `Science Fiction`, for example.

## Model Details

- **Model type:** BERT base uncased with a sequence-classification head
- **Task:** Multi-label text classification
- **Base model:** [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)
- **Number of labels:** 19
- **Maximum input length during training:** 256 tokens
- **Classification threshold used by the training script:** 0.5
- **Framework:** Hugging Face Transformers 5.5.0 and PyTorch

## Intended Use

This model is intended for experimentation, movie catalog organization, genre-based search, recommendation prototypes, and educational NLP projects. It predicts genres from textual movie metadata; it does not understand the full movie, watch its content, or provide content ratings.

## Labels

The model predicts the following genres:

`Adventure`, `War`, `Documentary`, `Science Fiction`, `Thriller`, `Family`, `Romance`, `Fantasy`, `Drama`, `Action`, `Music`, `Western`, `Animation`, `Mystery`, `TV Movie`, `Horror`, `Comedy`, `Crime`, and `History`.

## Input Format

Provide plain text containing a movie title, overview, or both. The training pipeline uses the `title_overview` field. Inputs are lowercased by the uncased BERT tokenizer and truncated to 256 tokens during training and inference.

## Usage

Install the required packages:

```bash
pip install torch transformers
```

Run inference with the model repository:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "prashantkhandelwal/movie-genres-classification"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

movie_text = "A crew travels through deep space to stop an alien threat from destroying Earth."
inputs = tokenizer(
	movie_text,
	max_length=256,
	truncation=True,
	return_tensors="pt",
)

with torch.no_grad():
	probabilities = torch.sigmoid(model(**inputs).logits)[0]

threshold = 0.5
predictions = [
	{
		"genre": model.config.id2label[index],
		"probability": round(float(probability), 4),
	}
	for index, probability in enumerate(probabilities)
	if probability >= threshold
]

print(predictions)
```

The threshold is a decision rule, not a calibrated probability guarantee. Applications may choose a threshold using a validation set and their preferred precision/recall tradeoff.

## Training

The model was fine-tuned with the following configuration:

- Dataset split: 95% training and 5% validation
- Split seed: `3407`
- Epochs: `3`
- Learning rate: `2e-5`
- Training batch size per device: `16`
- Evaluation batch size per device: `32`
- Weight decay: `0.01`
- Warmup ratio: `0.1`
- Evaluation and checkpoint saving: after each epoch
- Best checkpoint metric: validation micro F1
- Mixed precision: FP16 when supported by the training hardware

Genre names are converted to multi-hot vectors. The training objective treats each genre as an independent binary decision. Evaluation reports micro F1 and macro F1 at a threshold of 0.5.

## Evaluation

The training script computes:

- **Micro F1:** pools true positives, false positives, and false negatives across all genres before calculating F1.
- **Macro F1:** calculates F1 independently for each genre and averages the results.

Final evaluation scores are not included because they were not recorded as part of this model artifact. Run the evaluation cell in the accompanying training notebook or evaluate on a held-out dataset before relying on the model for a production workflow.

## Limitations and Biases

- Predictions depend on the quality, language, completeness, and style of the movie title and overview.
- The model was trained on movie metadata and may reproduce genre-labeling patterns or omissions in that data.
- Rare genres may receive less reliable predictions than common genres; inspect macro F1 and per-genre results.
- A fixed 0.5 threshold may not be optimal for every genre.
- Text longer than 256 tokens is truncated, which can remove useful plot information.
- The model is based on English BERT and should not be assumed to perform reliably on other languages.
- This model has not been validated for safety-critical, legal, or high-impact decision-making.

## Dataset and Provenance

The training pipeline uses the project's `cleaned_movies.csv`, containing movie title/overview text and comma-separated genre names. The repository includes database queries that extract movie metadata and genres, but the exact upstream dataset version, filtering history, and license are not recorded in the training script. Users should verify the source data's terms of use before redistributing or deploying the model.

## Files

The repository should contain the model weights and tokenizer files together:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin` (optional training metadata)

Checkpoint optimizer and random-state files are not required for inference.

## License

No model or dataset license is specified in the training project. Review the licenses of the base model and source movie metadata before using or redistributing this model.
