import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "moviedb.movies.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_movies.csv"
GENRES_FILE = DATA_DIR / "genres.csv"
OUTPUT_COLUMNS = [
    "overview",
    "genre_ids",
    "genre_names",
    "group_id",
    "split",
]
MISSING_VALUES = ["", "null", "none", "nan", "n/a", "na"]
PLACEHOLDER_PLOTS = [
    "add the plot.",
    "coming soon",
    "documentary film.",
    "overview coming soon...",
    "plot unknown.",
]
MIN_PLOT_CHARACTERS = 50
MIN_PLOT_WORDS = 8
MAX_GENRES = 6
RANDOM_SEED = 42


def iterative_multilabel_splits(
    label_matrix: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    if label_matrix.ndim != 2 or label_matrix.shape[0] < 10:
        raise ValueError("label_matrix must contain at least 10 rows")

    row_features = np.zeros((label_matrix.shape[0], 1), dtype=np.int8)
    train_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_seed,
    )
    train_indices, held_out_indices = next(
        train_splitter.split(row_features, label_matrix)
    )

    held_out_labels = label_matrix[held_out_indices]
    validation_test_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=random_seed,
    )
    validation_relative, test_relative = next(
        validation_test_splitter.split(
            row_features[held_out_indices], held_out_labels
        )
    )

    splits = np.empty(label_matrix.shape[0], dtype=object)
    splits[train_indices] = "train"
    splits[held_out_indices[validation_relative]] = "validation"
    splits[held_out_indices[test_relative]] = "test"
    return splits


def normalized_text(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def stable_group_id(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def assign_splits(frame: pl.DataFrame, genre_to_id: dict[str, int]) -> pl.DataFrame:
    genre_names = list(genre_to_id)
    label_matrix = np.asarray(
        [
            [int(genre_name in row_genres) for genre_name in genre_names]
            for row_genres in frame["genre_names_list"].to_list()
        ],
        dtype=np.int8,
    )
    return frame.with_columns(
        pl.Series("split", iterative_multilabel_splits(label_matrix))
    )


def clean_movies(
    input_file: Path,
    output_file: Path,
    genres_file: Path = GENRES_FILE,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    genre_reference = pl.read_csv(genres_file)
    genre_to_id = dict(
        genre_reference.select("genre_name", "genre_id").iter_rows()
    )
    raw = pl.read_csv(input_file, infer_schema_length=10_000)

    normalized = (
        raw.select(
            normalized_text("overview").alias("overview"),
            normalized_text("genres_name").alias("genre_names"),
        )
        .with_columns(
            pl.when(
                pl.col(column).is_null()
                | pl.col(column).str.to_lowercase().is_in(MISSING_VALUES)
            )
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ["overview", "genre_names"]
        )
        .with_columns(
            pl.col("overview").str.to_lowercase().alias("overview_key"),
        )
    )

    valid_text = normalized.filter(
        pl.col("overview").is_not_null()
        & pl.col("genre_names").is_not_null()
        & ~pl.col("overview_key").is_in(PLACEHOLDER_PLOTS)
        & ~pl.col("overview_key").str.starts_with("no overview")
        & ~pl.col("overview_key").str.starts_with(".....")
        & (pl.col("overview").str.len_chars() >= MIN_PLOT_CHARACTERS)
        & (pl.col("overview").str.count_matches(r"\S+") >= MIN_PLOT_WORDS)
    )

    exploded = (
        valid_text.with_columns(
            pl.col("genre_names").str.split(",").alias("genre_name"),
        )
        .explode("genre_name", empty_as_null=True)
        .with_columns(pl.col("genre_name").str.strip_chars())
    )
    unknown_genres = set(exploded["genre_name"].unique()) - set(genre_to_id)
    if unknown_genres:
        raise ValueError(f"Unknown genres in dataset: {sorted(unknown_genres)}")

    cleaned = (
        exploded.group_by("overview_key")
        .agg(
            pl.col("overview").first(),
            pl.col("genre_name").unique().sort().alias("genre_names_list"),
        )
        .filter(
            pl.col("genre_names_list").list.len().is_between(1, MAX_GENRES)
        )
        .with_columns(
            pl.col("overview_key")
            .map_elements(stable_group_id, return_dtype=pl.UInt64)
            .alias("group_id"),
            pl.col("genre_names_list")
            .list.eval(pl.element().replace_strict(genre_to_id))
            .alias("genre_ids_list"),
        )
        .with_columns(
            pl.col("genre_names_list").list.join(",").alias("genre_names"),
            pl.col("genre_ids_list")
            .list.eval(pl.element().cast(pl.String))
            .list.join(",")
            .alias("genre_ids"),
        )
        .sort("overview_key")
        .pipe(assign_splits, genre_to_id)
        .select(OUTPUT_COLUMNS)
    )

    cleaned.write_csv(output_file)
    print(f"Rows: {raw.height:,} raw -> {cleaned.height:,} cleaned")
    print(cleaned.group_by("split").len().sort("split").to_dicts())


def main() -> None:
    clean_movies(INPUT_FILE, OUTPUT_FILE)
    print(f"Cleaned data written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
