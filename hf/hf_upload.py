import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def main() -> None:
	load_dotenv(Path(__file__).with_name(".env"))

	parser = argparse.ArgumentParser(
		description="Upload a model directory to the Hugging Face Hub."
	)
	parser.add_argument(
		"directory",
		nargs="?",
		type=Path,
		default=os.getenv("HF_UPLOAD_DIR", "outputs/bert-movie-genres"),
		help="Local model directory to upload",
	)
	parser.add_argument("repo_id", nargs="?", default=os.getenv("HF_REPO_ID"))
	parser.add_argument(
		"--repo-type",
		choices=("model", "dataset", "space"),
		default=os.getenv("HF_REPO_TYPE", "model"),
	)
	parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
	parser.add_argument(
		"--commit-message",
		default=os.getenv("HF_COMMIT_MESSAGE", "Upload file"),
	)
	args = parser.parse_args()

	if args.repo_id is None:
		parser.error("Repository ID is required (argument or HF_REPO_ID in .env)")
	if not args.directory.is_dir():
		parser.error(f"Directory not found: {args.directory}")

	result = HfApi(token=args.token).upload_folder(
		folder_path=str(args.directory),
		repo_id=args.repo_id,
		repo_type=args.repo_type,
		commit_message=args.commit_message,
		# Checkpoints contain optimizer state and are not needed for inference.
		ignore_patterns=["checkpoint-*", "checkpoint-*/*"],
	)
	print(result)


if __name__ == "__main__":
	main()
