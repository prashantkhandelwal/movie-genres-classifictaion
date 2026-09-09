MAX_LENGTH = 256


def build_model_text(overview: str) -> str:
    return f"Overview: {overview.strip()}"