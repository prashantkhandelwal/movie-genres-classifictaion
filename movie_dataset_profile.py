from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

path = Path(r"f:\Github\movie-genres-classifictaion\data\moviedb.movies.csv")
genres_path = path.with_name("genres.csv")
df = pl.read_csv(path, infer_schema_length=10000)

print("FILE_MB", round(path.stat().st_size / 1024**2, 2))
print("SHAPE", df.shape)
print("COLUMNS", df.columns)
for name in df.columns:
    series = df[name].cast(pl.String, strict=False)
    normalized = series.str.strip_chars().str.to_lowercase()
    null_like = (
        series.is_null()
        | normalized.is_in(["", "null", "none", "nan", "n/a", "na"])
    ).sum()
    print("COLUMN", name, "NULL_LIKE", null_like, "N_UNIQUE", series.n_unique())

plot = df["overview"].cast(pl.String, strict=False).fill_null("").str.strip_chars()
plot_lower = plot.str.to_lowercase()
chars = plot.str.len_chars()
words = plot.str.split(" ").list.len()
quantiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
print("PLOT_CHAR_QUANTILES", {str(q): chars.quantile(q, interpolation="nearest") for q in quantiles})
print("PLOT_WORD_QUANTILES", {str(q): words.quantile(q, interpolation="nearest") for q in quantiles})
for threshold in [20, 50, 100, 200]:
    print("PLOT_CHARS_LT", threshold, int((chars < threshold).sum()))
for marker in ["no overview found.", "no overview", "....."]:
    print("PLOT_MARKER", repr(marker), int(plot_lower.str.starts_with(marker).sum()))
print("EMPTY_PLOT", int((plot == "").sum()))

label_sets = []
id_sets = []
labels = []
for names, ids in zip(
    df["genres_name"].cast(pl.String, strict=False).to_list(),
    df["genres_id"].cast(pl.String, strict=False).to_list(),
):
    names_tuple = tuple(sorted(value.strip() for value in (names or "").split(",") if value.strip()))
    ids_tuple = tuple(value.strip() for value in (ids or "").split(",") if value.strip())
    label_sets.append(names_tuple)
    id_sets.append(ids_tuple)
    labels.extend(names_tuple)

print("LABEL_CARDINALITY", dict(sorted(Counter(map(len, label_sets)).items())))
print("LABEL_FREQUENCY", dict(Counter(labels).most_common()))
print("GENRE_NAME_ID_LENGTH_MISMATCH", sum(len(names) != len(ids) for names, ids in zip(label_sets, id_sets)))
expected = set(pl.read_csv(genres_path)["genre_name"].to_list())
print("UNKNOWN_LABELS", sorted(set(labels) - expected))
print("MISSING_EXPECTED_LABELS", sorted(expected - set(labels)))

plots = plot.to_list()
pair_counts = Counter(zip(plots, label_sets))
text_counts = Counter(plots)
labels_by_text = defaultdict(set)
for text, genre_set in zip(plots, label_sets):
    labels_by_text[text].add(genre_set)
conflicting_texts = {text for text, sets in labels_by_text.items() if len(sets) > 1}
print("DUP_OVERVIEW_EXTRA_ROWS", sum(count - 1 for count in text_counts.values()))
print("EXACT_TEXT_LABEL_DUP_EXTRA_ROWS", sum(count - 1 for count in pair_counts.values()))
print("TEXTS_WITH_CONFLICTING_LABEL_SETS", len(conflicting_texts))
print("ROWS_IN_CONFLICT_GROUPS", sum(text_counts[text] for text in conflicting_texts))
print("TOP_REPEATED_TEXTS", [(text[:100], count, len(labels_by_text[text])) for text, count in text_counts.most_common(10)])

for column in ["title", "original_title"]:
    if column in df.columns:
        values = df[column].cast(pl.String, strict=False).fill_null("").str.strip_chars().str.to_lowercase()
        print("DUPLICATE_EXTRA_ROWS", column, df.height - values.n_unique())
