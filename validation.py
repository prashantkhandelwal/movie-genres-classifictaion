from pathlib import Path

import polars as pl

path = Path(r"f:\Github\movie-genres-classifictaion\data\cleaned_movies.csv")
df = pl.read_csv(path)
labels = df.with_columns(pl.col("genre_names").str.split(",").alias("label")).explode("label")
plot_lower = pl.col("plot").str.to_lowercase()

assert df.columns == ["plot", "genre_ids", "genre_names", "group_id", "split"]
assert df["plot"].n_unique() == df.height
assert df.group_by("group_id").agg(pl.col("split").n_unique().alias("splits"))["splits"].max() == 1
assert df["split"].null_count() == 0
assert set(df["split"].unique()) == {"train", "validation", "test"}
assert int(df.select(plot_lower.str.contains("overview: no overview").sum()).item()) == 0
assert int(df.select(plot_lower.str.contains("overview: coming soon").sum()).item()) == 0
assert int(df.select(pl.col("genre_names").str.split(",").list.len().max()).item()) <= 6
assert set(labels["label"].unique()) == set(pl.read_csv(path.with_name("genres.csv"))["genre_name"])

support = labels.group_by("split", "label").len().sort("label", "split")
wide = support.pivot(on="split", index="label", values="len")
for split in ["train", "validation", "test"]:
    wide = wide.with_columns((pl.col(split) / pl.col(split).sum()).alias(f"{split}_share"))
max_gap = wide.select(
    pl.max_horizontal("train_share", "validation_share", "test_share")
    - pl.min_horizontal("train_share", "validation_share", "test_share")
).max().item()

print("rows", df.height)
print("file_mb", round(path.stat().st_size / 1024**2, 2))
print("splits", df.group_by("split").len().sort("split").to_dicts())
print("duplicate_plots", df.height - df["plot"].n_unique())
print("cross_split_groups", 0)
print("labels", labels["label"].n_unique())
print("max_genres", df.select(pl.col("genre_names").str.split(",").list.len().max()).item())
print("max_label_prevalence_gap", round(max_gap, 6))
print("label_support", support.to_dicts())
