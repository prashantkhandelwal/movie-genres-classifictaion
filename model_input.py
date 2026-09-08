MAX_LENGTH = 256


def build_training_text(title: str, overview: str, keywords: str) -> str:
    return (
        f"Title: {title.strip()} "
        f"Keywords: {keywords.strip()} "
        f"Overview: {overview.strip()}"
    )


def build_inference_text(overview: str) -> str:
    return f"Overview: {overview.strip()}"