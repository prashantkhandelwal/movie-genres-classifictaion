# Automated Model Releases

The `Train and release model` GitHub Actions workflow performs the complete release sequence:

1. Restores prior training history from the Hugging Face model repository.
2. Trains the model on a self-hosted Windows GPU runner.
3. Generates comparison and release notes.
4. Uploads the latest model and retained history to Hugging Face.
5. Tags the Hugging Face revision as `model-<run-id>`.
6. Creates a GitHub Release with metrics, thresholds, plots, and comparison reports.

Publishing steps run only after training completes successfully. Model weights are hosted on Hugging Face rather than attached to GitHub Releases because GitHub release assets have size limits.

## One-Time Setup

### Hugging Face

Create the destination model repository on Hugging Face, then create a fine-grained access token with write access to that repository.

In the GitHub repository, open **Settings > Secrets and variables > Actions** and add:

- Repository secret `HF_TOKEN`: the Hugging Face write token.
- Repository variable `HF_REPO_ID`: the model repository ID, such as `username/movie-genres-classification`.

### GPU Runner

In GitHub, open **Settings > Actions > Runners**, add a self-hosted Windows runner, and assign it the custom label `gpu`. The runner must have:

- An NVIDIA GPU and compatible driver.
- Git installed.
- Network access to GitHub, PyPI, and Hugging Face.
- Enough disk space for dependencies, checkpoints, and model artifacts.

The workflow installs `uv` and project dependencies itself. It verifies CUDA before starting training and fails early if the GPU is unavailable.

### GitHub Permissions

Under **Settings > Actions > General > Workflow permissions**, allow GitHub Actions to create repository contents. The workflow requests `contents: write` for its generated tag and release.

## Run a Release

Open **Actions > Train and release model > Run workflow**. The workflow has a concurrency lock, so only one training release runs at a time.

Successful releases appear in two places:

- Hugging Face: latest model files plus immutable `model-<run-id>` tags.
- GitHub Releases: report assets and release notes tied to the same tag.

If publishing fails after training, the workflow uploads the run summary and history as a temporary GitHub Actions artifact for recovery. It does not create a GitHub Release for failed or interrupted training runs.