from pathlib import Path

import polars as pl


DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "moviedb.movies.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_movies.csv"
OUTPUT_COLUMNS = ["plot", "genre_ids", "genre_names"]
MISSING_VALUES = ["", "null", "none", "nan", "n/a", "na"]


def clean_movies(input_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cleaned = (
        pl.scan_csv(input_file)
        .select(
            pl.col("overview").cast(pl.String, strict=False).alias("plot"),
            pl.col("genres_id").cast(pl.String, strict=False).alias("genre_ids"),
            pl.col("genres_name").cast(pl.String, strict=False).alias("genre_names"),
        )
        .with_columns(
            pl.col(column).str.strip_chars()
            for column in OUTPUT_COLUMNS
        )
        .filter(
            pl.all_horizontal(
                pl.col(column).is_not_null()
                & ~pl.col(column).str.to_lowercase().is_in(MISSING_VALUES)
                for column in OUTPUT_COLUMNS
            )
            & ~pl.col("plot").str.starts_with(".....")
        )
    )

    cleaned.sink_csv(output_file)


def main() -> None:
    clean_movies(INPUT_FILE, OUTPUT_FILE)
    print(f"Cleaned data written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
