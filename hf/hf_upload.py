import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


ROOT_DIR = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_FILES = (
	"config.json",
	"model.safetensors",
	"tokenizer.json",
	"tokenizer_config.json",
)


def repository_path(value: str | Path) -> Path:
	path = Path(value).expanduser()
	if not path.is_absolute():
		path = ROOT_DIR / path
	return path.resolve()


def main() -> None:
	load_dotenv(ROOT_DIR / ".env")

	parser = argparse.ArgumentParser(
		description="Upload a model directory to the Hugging Face Hub."
	)
	parser.add_argument(
		"directory",
		nargs="?",
		type=repository_path,
		default=repository_path(
			os.getenv("HF_UPLOAD_DIR", "outputs/bert-movie-genres")
		),
		help="Model directory to upload; relative paths use the repository root",
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
	missing_files = [
		filename
		for filename in REQUIRED_MODEL_FILES
		if not (args.directory / filename).is_file()
	]
	if missing_files:
		parser.error("Missing required model files: " + ", ".join(missing_files))

	result = HfApi(token=args.token).upload_folder(
		folder_path=str(args.directory),
		repo_id=args.repo_id,
		repo_type=args.repo_type,
		commit_message=args.commit_message,
		allow_patterns=list(REQUIRED_MODEL_FILES),
	)
	print(result)


if __name__ == "__main__":
	main()
