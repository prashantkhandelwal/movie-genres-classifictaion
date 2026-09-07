MAX_LENGTH = 256


def build_model_text(title: str, overview: str, keywords: str) -> str:
    return (
        f"Title: {title.strip()} "
        f"Keywords: {keywords.strip()} "
        f"Overview: {overview.strip()}"
    )