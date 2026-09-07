from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="prashantkhandelwal/movie-genres-classification",
    top_k=None,
)

result = classifier(
    "Title: Wish You Were Here Keywords: romance,dark wish "
    "Overview: After breaking the mysterious 'One Wish Willow' to win his "
    "crush's heart, a hopeless romantic finds himself getting exactly what "
    "he asked for but soon discovers that some desires come at a dark, "
    "sinister price.",
    max_length=256,
    truncation=True,
)

genres = [label for label in result[0] if label["score"] >= 0.5]
print(genres)