"""Shared constants for data loading, training, and evaluation.

Centralizes label definitions so the loader, training script, and
evaluation code never disagree about what the model is trained on.
"""

# The ten labels the classifier is trained on, sorted alphabetically.
# This order must match the column order of the model's predict_proba
# output (sklearn sorts classes_ lexicographically for string labels).
TRAINED_LABELS: list[str] = [
    "business",
    "entertainment",
    "food",
    "graphics",
    "historical",
    "medical",
    "politics",
    "space",
    "sport",
    "technology",
]

# Fallback label for low-confidence / out-of-distribution documents.
# Never trained on; assigned only by threshold routing in src.predict.
FALLBACK_LABEL: str = "other"

# Folder names on disk that must be normalized to a canonical label.
# The dataset ships with a "technologie" folder which is a typo for
# "technology" — folder names are mapped through this table.
LABEL_ALIASES: dict[str, str] = {"technologie": "technology"}

# Folders excluded from the training pool. "other" is kept aside as an
# out-of-distribution holdout used only after training.
EXCLUDED_TRAINING_LABELS: list[str] = [FALLBACK_LABEL]

# Default seed for every stochastic step (splits, models).
RANDOM_STATE: int = 42
