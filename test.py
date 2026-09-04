import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = Path(__file__).with_name("outputs") / "bert-movie-genres"
MAX_LENGTH = 256
DEFAULT_THRESHOLD = 0.1


def predict_genres(
	title: str,
	overview: str,
	threshold: float = DEFAULT_THRESHOLD,
) -> list[tuple[str, float]]:
	if not MODEL_PATH.is_dir():
		raise FileNotFoundError(
			f"Trained model not found at {MODEL_PATH}. Run train.py first."
		)
	if not 0.0 <= threshold <= 1.0:
		raise ValueError("Threshold must be between 0 and 1.")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
	model.to(device)
	model.eval()

	text = f"{title.strip()} - {overview.strip()}"
	inputs = tokenizer(
		text,
		return_tensors="pt",
		max_length=MAX_LENGTH,
		truncation=True,
	).to(device)

	with torch.inference_mode():
		probabilities = torch.sigmoid(model(**inputs).logits)[0].cpu()

	predictions = [
		(model.config.id2label[index], float(probability))
		for index, probability in enumerate(probabilities)
		if probability >= threshold
	]
	return sorted(predictions, key=lambda prediction: prediction[1], reverse=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Predict one or more genres for a movie."
	)
	parser.add_argument("title", nargs="?", help="Movie title")
	parser.add_argument("overview", nargs="?", help="Movie overview or plot summary")
	parser.add_argument(
		"--threshold",
		type=float,
		default=DEFAULT_THRESHOLD,
		help="Minimum genre probability (default: 0.5)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	title = args.title or input("Movie title: ").strip()
	overview = args.overview or input("Movie overview: ").strip()

	predictions = predict_genres(title, overview, args.threshold)
	if not predictions:
		print(f"No genre reached the {args.threshold:.0%} threshold.")
		return

	print("Predicted genres:")
	for genre, probability in predictions:
		print(f"- {genre}: {probability:.1%}")


if __name__ == "__main__":
	main()
