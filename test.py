import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_input import MAX_LENGTH, build_model_text


MODEL_PATH = Path(__file__).with_name("outputs") / "bert-movie-genres" / "checkpoint-26202"
THRESHOLDS_PATH = Path(__file__).with_name("outputs") / "bert-movie-genres" / "thresholds.json"
DEFAULT_THRESHOLD = 0.5


def load_genre_thresholds() -> dict[str, float]:
	if not THRESHOLDS_PATH.is_file():
		return {}
	threshold_data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
	return {
		genre: float(threshold)
		for genre, threshold in threshold_data["per_label_thresholds"].items()
	}


def predict_genres(
	title: str,
	overview: str,
	threshold: float | None = None,
	*,
	keywords: str = "",
) -> list[tuple[str, float]]:
	if not MODEL_PATH.is_dir():
		raise FileNotFoundError(
			f"Trained model not found at {MODEL_PATH}. Run train.py first."
		)
	if threshold is not None and not 0.0 <= threshold <= 1.0:
		raise ValueError("Threshold must be between 0 and 1.")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
	model.to(device)
	model.eval()

	text = build_model_text(title, overview, keywords)
	inputs = tokenizer(
		text,
		return_tensors="pt",
		max_length=MAX_LENGTH,
		truncation=True,
	).to(device)

	with torch.inference_mode():
		probabilities = torch.sigmoid(model(**inputs).logits)[0].cpu()

	genre_thresholds = load_genre_thresholds()
	predictions = [
		(model.config.id2label[index], float(probability))
		for index, probability in enumerate(probabilities)
		if probability
		>= (
			threshold
			if threshold is not None
			else genre_thresholds.get(
				model.config.id2label[index], DEFAULT_THRESHOLD
			)
		)
	]
	return sorted(predictions, key=lambda prediction: prediction[1], reverse=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Predict one or more genres for a movie."
	)
	parser.add_argument("title", nargs="?", help="Movie title")
	parser.add_argument("overview", nargs="?", help="Movie overview or plot summary")
	parser.add_argument(
		"--keywords",
		default="",
		help="Comma-separated movie keywords",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=None,
		help="Override all saved per-genre thresholds with one value",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	title = args.title or input("Movie title: ").strip()
	overview = args.overview or input("Movie overview: ").strip()

	predictions = predict_genres(
		title,
		overview,
		args.threshold,
		keywords=args.keywords,
	)
	if not predictions:
		if args.threshold is None:
			print("No genre reached its configured threshold.")
		else:
			print(f"No genre reached the {args.threshold:.0%} threshold.")
		return

	print("Predicted genres:")
	for genre, probability in predictions:
		print(f"- {genre}: {probability:.1%}")


if __name__ == "__main__":
	main()
