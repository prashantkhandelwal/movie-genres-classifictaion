from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

path = Path(r"f:\Github\movie-genres-classifictaion\data\cleaned_movies.csv")
plot_path = path.parent.parent / "result" / "validation_distribution.png"
split_names = ["train", "validation", "test"]
df = pl.read_csv(path)
labels = df.with_columns(pl.col("genre_names").str.split(",").alias("label")).explode(
    "label", empty_as_null=True
)
overview_lower = pl.col("overview").str.to_lowercase()

assert df.columns == [
    "overview",
    "genre_ids",
    "genre_names",
    "group_id",
    "split",
]
assert df["overview"].n_unique() == df.height
assert df.group_by("group_id").agg(pl.col("split").n_unique().alias("splits"))["splits"].max() == 1
assert df["split"].null_count() == 0
assert set(df["split"].unique()) == {"train", "validation", "test"}
assert int(df.select(overview_lower.str.starts_with("no overview").sum()).item()) == 0
assert int(df.select(overview_lower.str.starts_with("coming soon").sum()).item()) == 0
assert int(df.select(pl.col("genre_names").str.split(",").list.len().max()).item()) <= 6
assert set(labels["label"].unique()) == set(pl.read_csv(path.with_name("genres.csv"))["genre_name"])

support = labels.group_by("split", "label").len().sort("label", "split")
wide = support.pivot(on="split", index="label", values="len")
split_sizes = {
    split_name: df.filter(pl.col("split") == split_name).height
    for split_name in split_names
}
for split_name in split_names:
    wide = wide.with_columns(
        (pl.col(split_name) / split_sizes[split_name]).alias(
            f"{split_name}_share"
        )
    )
max_gap = wide.select(
    pl.max_horizontal("train_share", "validation_share", "test_share")
    - pl.min_horizontal("train_share", "validation_share", "test_share")
).max().item()
assert max_gap <= 0.005

plot_path.parent.mkdir(parents=True, exist_ok=True)
figure, (size_axis, prevalence_axis) = plt.subplots(
    1,
    2,
    figsize=(16, 10),
    gridspec_kw={"width_ratios": [1, 2.5]},
)
colors = ["#287271", "#d9895b", "#355c9a"]
size_values = [split_sizes[split_name] for split_name in split_names]
size_axis.bar(split_names, size_values, color=colors)
size_axis.set_title("Movies per Split")
size_axis.set_ylabel("Movies")
size_axis.grid(axis="y", alpha=0.25)
for index, value in enumerate(size_values):
    size_axis.text(index, value, f"{value:,}", ha="center", va="bottom")

wide = wide.sort("train_share")
genre_positions = np.arange(wide.height)
bar_height = 0.25
for offset, (split_name, color) in enumerate(zip(split_names, colors)):
    prevalence_axis.barh(
        genre_positions + (offset - 1) * bar_height,
        wide[f"{split_name}_share"].to_numpy() * 100,
        height=bar_height,
        label=split_name.title(),
        color=color,
    )
prevalence_axis.set_yticks(genre_positions, wide["label"].to_list())
prevalence_axis.set_title("Genre Prevalence by Split")
prevalence_axis.set_xlabel("Movies with genre (%)")
prevalence_axis.grid(axis="x", alpha=0.25)
prevalence_axis.legend()
figure.suptitle("Dataset Validation Distribution", fontsize=16)
figure.tight_layout()
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print("rows", df.height)
print("file_mb", round(path.stat().st_size / 1024**2, 2))
print("splits", df.group_by("split").len().sort("split").to_dicts())
print("duplicate_overviews", df.height - df["overview"].n_unique())
print("cross_split_groups", 0)
print("labels", labels["label"].n_unique())
print("max_genres", df.select(pl.col("genre_names").str.split(",").list.len().max()).item())
print("max_label_prevalence_gap", round(max_gap, 6))
print("label_support", support.to_dicts())
print("plot", plot_path)
