"""Train the document classifier and save the model bundle.

The script is intentionally boring: it recreates the fixed 70/15/15
split after dataset-level deduplication, trains the selected TF-IDF
pipeline on the training split only, and writes the contract-compatible
joblib bundle consumed by the API and ``src.predict``.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    EXCLUDED_TRAINING_LABELS,
    LABEL_ALIASES,
    RANDOM_STATE,
    TRAINED_LABELS,
)
from src.data_loader import DatasetBundle, load_dataset

DEFAULT_THRESHOLDS = {
    "auto_accept": 0.90,
    "manual_review": 0.70,
    "other": 0.55,
}


def build_model(model_name: str) -> Pipeline:
    """Create one of the candidate pipelines from the experiment notebook."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "multinomial_nb":
        classifier = MultinomialNB()
    elif model_name == "calibrated_linear_svc":
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def stratified_split(
    texts: list[str],
    labels: list[str],
    paths: list[str],
    val_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, dict[str, list[str]]]:
    """Return deterministic stratified train/validation/test splits."""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("--val-size and --test-size must be >0 and sum to less than 1")

    train_val_texts, test_texts, train_val_labels, test_labels, train_val_paths, test_paths = (
        train_test_split(
            texts,
            labels,
            paths,
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
    )

    adjusted_val_size = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels, train_paths, val_paths = train_test_split(
        train_val_texts,
        train_val_labels,
        train_val_paths,
        test_size=adjusted_val_size,
        stratify=train_val_labels,
        random_state=random_state,
    )

    return {
        "train": {
            "texts": [str(value) for value in train_texts],
            "labels": [str(value) for value in train_labels],
            "paths": [str(value) for value in train_paths],
        },
        "validation": {
            "texts": [str(value) for value in val_texts],
            "labels": [str(value) for value in val_labels],
            "paths": [str(value) for value in val_paths],
        },
        "test": {
            "texts": [str(value) for value in test_texts],
            "labels": [str(value) for value in test_labels],
            "paths": [str(value) for value in test_paths],
        },
    }


def make_bundle(model: Pipeline, model_type: str, target_names: list[str]) -> dict[str, Any]:
    """Wrap a fitted sklearn pipeline in the frozen model-bundle shape."""
    return {
        "model": model,
        "target_names": target_names,
        "model_type": model_type,
        "trained_labels": TRAINED_LABELS,
        "excluded_training_labels": EXCLUDED_TRAINING_LABELS,
        "label_aliases": LABEL_ALIASES,
        "confidence_thresholds": DEFAULT_THRESHOLDS,
    }


def write_training_summary(
    dataset: DatasetBundle,
    splits: dict[str, dict[str, list[str]]],
    args: argparse.Namespace,
) -> None:
    """Persist split counts and loader diagnostics for reproducibility."""
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "random_state": args.random_state,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "known_documents_after_dedup": len(dataset.known_texts),
        "other_holdout_documents": len(dataset.other_texts),
        "duplicate_report": dataset.duplicate_report,
        "empty_files": dataset.empty_files,
        "split_counts": {
            split: {
                "total": len(values["texts"]),
                "by_label": {label: values["labels"].count(label) for label in TRAINED_LABELS},
            }
            for split, values in splits.items()
        },
    }
    (report_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the document classifier")
    parser.add_argument("--data-dir", required=True, help="Path to trellis_assessment_ds")
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "multinomial_nb", "calibrated_linear_svc"],
        help="Candidate model pipeline to train",
    )
    parser.add_argument("--output", default="models/document_classifier.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    """Train and serialize the model bundle."""
    args = parse_args()
    dataset = load_dataset(args.data_dir)
    splits = stratified_split(
        dataset.known_texts,
        dataset.known_labels,
        dataset.known_paths,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_model(args.model)
    model.fit(splits["train"]["texts"], splits["train"]["labels"])

    target_names = [str(label) for label in model.classes_]
    if target_names != TRAINED_LABELS:
        raise ValueError(f"Model classes {target_names} do not match {TRAINED_LABELS}")

    bundle = make_bundle(model, args.model, target_names)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)

    write_training_summary(dataset, splits, args)
    print(f"Saved model bundle to {output}")


if __name__ == "__main__":
    main()
