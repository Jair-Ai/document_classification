"""Tests for the error-analysis report writers in ``src.evaluate``.

These artifacts (``misclassified_examples.csv`` and
``other_holdout_predictions.csv``) are consumed by
``notebooks/03_error_analysis.ipynb``. The tests pin their exact column
schema and core logic so the notebook never silently breaks and the
reports stay reproducible from a clean clone.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluate import (
    write_misclassified_examples,
    write_other_holdout_predictions,
)
from src.predict import predict_batch

# Column order the notebook reads back; must not drift.
_MISCLASSIFIED_COLUMNS = [
    "path",
    "actual_label",
    "raw_label",
    "final_label",
    "confidence",
    "decision",
    "top_k",
    "text_excerpt",
]
_OTHER_HOLDOUT_COLUMNS = [
    "path",
    "final_label",
    "raw_label",
    "confidence",
    "decision",
    "top_k",
    "text_excerpt",
]

_TEXTS = [
    "Quarterly profits surged as the company beat revenue forecasts",
    "The striker scored twice to win the championship final",
    "The new smartphone ships with a faster chip and better camera",
]


def test_misclassified_examples_schema_and_filter(
    fixture_bundle: dict[str, Any], tmp_path: Path
) -> None:
    """Only argmax-wrong rows are written, with the frozen column schema."""
    raw = [pred["raw_label"] for pred in predict_batch(_TEXTS, fixture_bundle, top_k=3)]
    other_labels = list(fixture_bundle["trained_labels"])

    # Row 0 labeled correctly (excluded); rows 1-2 deliberately mislabeled.
    actual = [raw[0]] + [next(label for label in other_labels if label != r) for r in raw[1:]]
    paths = [f"doc_{i}.txt" for i in range(len(_TEXTS))]

    write_misclassified_examples(fixture_bundle, _TEXTS, actual, paths, tmp_path)

    df = pd.read_csv(tmp_path / "misclassified_examples.csv")
    assert list(df.columns) == _MISCLASSIFIED_COLUMNS
    # The correctly-labeled row must be filtered out.
    assert set(df["path"]) == {"doc_1.txt", "doc_2.txt"}
    # Every written row is genuinely misclassified.
    assert (df["raw_label"] != df["actual_label"]).all()
    # top_k round-trips as JSON.
    parsed = json.loads(df.iloc[0]["top_k"])
    assert {"label", "confidence"} == set(parsed[0])


def test_misclassified_examples_empty_when_all_correct(
    fixture_bundle: dict[str, Any], tmp_path: Path
) -> None:
    """A header-only file is written when nothing is misclassified."""
    raw = [pred["raw_label"] for pred in predict_batch(_TEXTS, fixture_bundle, top_k=3)]
    paths = [f"doc_{i}.txt" for i in range(len(_TEXTS))]

    write_misclassified_examples(fixture_bundle, _TEXTS, raw, paths, tmp_path)

    df = pd.read_csv(tmp_path / "misclassified_examples.csv")
    assert list(df.columns) == _MISCLASSIFIED_COLUMNS
    assert df.empty


def test_other_holdout_predictions_schema_and_completeness(
    fixture_bundle: dict[str, Any], tmp_path: Path
) -> None:
    """One row per holdout document, with the frozen column schema."""
    paths = [f"other_{i}.txt" for i in range(len(_TEXTS))]

    write_other_holdout_predictions(fixture_bundle, _TEXTS, paths, tmp_path)

    df = pd.read_csv(tmp_path / "other_holdout_predictions.csv")
    assert list(df.columns) == _OTHER_HOLDOUT_COLUMNS
    assert list(df["path"]) == paths  # every document is written, in order
    assert df["confidence"].between(0.0, 1.0).all()
