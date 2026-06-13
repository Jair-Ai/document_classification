"""Evaluate a trained document-classifier bundle.

Reports are intentionally file-based so the final README can quote the
same measured artifacts reviewers can inspect locally.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.confidence_analysis import (
    choose_other_threshold,
    confidence_range_report,
    threshold_tradeoff_table,
    threshold_tradeoffs,
    top_confidences,
)
from src.config import RANDOM_STATE, TRAINED_LABELS
from src.data_loader import load_dataset
from src.predict import predict_text
from src.train import stratified_split


def raw_predictions(bundle: dict[str, Any], texts: list[str]) -> list[str]:
    """Return raw model labels, before threshold routing."""
    return [str(label) for label in bundle["model"].predict(texts)]


def write_classification_reports(
    bundle: dict[str, Any],
    test_texts: list[str],
    test_labels: list[str],
    report_dir: Path,
) -> None:
    """Write test-set classification report and confusion matrix."""
    predictions = raw_predictions(bundle, test_texts)
    text_report = cast(
        str,
        classification_report(
            test_labels,
            predictions,
            labels=TRAINED_LABELS,
            digits=4,
            zero_division="warn",
        ),
    )
    json_report = cast(
        dict[str, Any],
        classification_report(
            test_labels,
            predictions,
            labels=TRAINED_LABELS,
            output_dict=True,
            zero_division="warn",
        ),
    )
    matrix = confusion_matrix(test_labels, predictions, labels=TRAINED_LABELS)

    (report_dir / "classification_report.txt").write_text(text_report, encoding="utf-8")
    (report_dir / "classification_report.json").write_text(
        json.dumps(json_report, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(matrix, index=TRAINED_LABELS, columns=TRAINED_LABELS).to_csv(
        report_dir / "confusion_matrix.csv",
        index_label="actual",
    )


def write_confidence_ranges(
    bundle: dict[str, Any],
    val_texts: list[str],
    val_labels: list[str],
    report_dir: Path,
) -> None:
    """Write validation accuracy by confidence band."""
    predictions = raw_predictions(bundle, val_texts)
    confidences = top_confidences(bundle, val_texts)
    correct = pd.Series(predictions).eq(pd.Series(val_labels)).to_numpy()
    confidence_range_report(confidences, correct).to_csv(
        report_dir / "confidence_ranges.csv",
        index=False,
    )


def write_other_holdout_report(
    bundle: dict[str, Any],
    other_texts: list[str],
    other_paths: list[str],
    report_dir: Path,
) -> None:
    """Run the fallback policy on the OOD holdout and summarize routing."""
    predictions = [
        {"path": path, **predict_text(text, bundle, top_k=3)}
        for path, text in zip(other_paths, other_texts, strict=True)
    ]
    routed_to_other = sum(1 for pred in predictions if pred["label"] == "other")
    total = len(predictions)
    report = {
        "total": total,
        "routed_to_other": routed_to_other,
        "routed_to_other_pct": round(routed_to_other / total, 4) if total else None,
        "not_routed_to_other": total - routed_to_other,
        "not_routed_to_other_pct": round((total - routed_to_other) / total, 4) if total else None,
        "predictions": predictions,
    }
    (report_dir / "other_holdout_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def write_inference_benchmark(
    bundle: dict[str, Any],
    model_path: Path,
    texts: list[str],
    report_dir: Path,
    repeats: int = 200,
) -> None:
    """Measure single-document latency and batch throughput."""
    model = bundle["model"]
    sample = texts[0]
    model.predict_proba([sample])

    laps = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba([sample])
        laps.append((time.perf_counter() - start) * 1000)

    batch = texts[: min(len(texts), 256)]
    start = time.perf_counter()
    model.predict_proba(batch)
    elapsed = time.perf_counter() - start

    benchmark = {
        "single_doc_latency_ms_median": round(statistics.median(laps), 4),
        "single_doc_latency_ms_p95": round(statistics.quantiles(laps, n=20)[18], 4),
        "batch_size": len(batch),
        "batch_docs_per_second": round(len(batch) / elapsed, 2),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
    }
    (report_dir / "inference_benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n",
        encoding="utf-8",
    )


def tune_and_save_thresholds(
    bundle: dict[str, Any],
    model_path: Path,
    val_texts: list[str],
    test_texts: list[str],
    other_texts: list[str],
    report_dir: Path,
) -> dict[str, Any]:
    """Choose the fallback threshold on validation and persist it."""
    validation_rows = threshold_tradeoffs(bundle, val_texts, other_texts)
    choice = choose_other_threshold(validation_rows)

    thresholds = dict(bundle.get("confidence_thresholds", {}))
    thresholds.update(
        {
            "auto_accept": 0.90,
            "manual_review": 0.70,
            "other": choice.other_threshold,
        }
    )
    bundle["confidence_thresholds"] = thresholds
    joblib.dump(bundle, model_path)

    threshold_tradeoff_table(validation_rows).to_csv(
        report_dir / "threshold_tradeoff_validation.csv",
        index=False,
    )
    threshold_tradeoff_table(threshold_tradeoffs(bundle, test_texts, other_texts)).to_csv(
        report_dir / "threshold_tradeoff_test.csv",
        index=False,
    )
    selection = {
        "confidence_thresholds": thresholds,
        "selection_basis": "validation known-label misroute guardrail plus anecdotal OOD holdout",
        "reason": choice.reason,
        "other_holdout_note": (
            "The OOD side has only six files, so it is useful as a smoke "
            "check but too small for a statistically stable threshold."
        ),
    }
    (report_dir / "threshold_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained document classifier")
    parser.add_argument("--data-dir", required=True, help="Path to trellis_assessment_ds")
    parser.add_argument("--model-path", default="models/document_classifier.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    """Generate all evaluation reports and update bundle thresholds."""
    args = parse_args()
    model_path = Path(args.model_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = joblib.load(model_path)
    dataset = load_dataset(args.data_dir)
    splits = stratified_split(
        dataset.known_texts,
        dataset.known_labels,
        dataset.known_paths,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    write_classification_reports(
        bundle,
        splits["test"]["texts"],
        splits["test"]["labels"],
        report_dir,
    )
    write_confidence_ranges(
        bundle,
        splits["validation"]["texts"],
        splits["validation"]["labels"],
        report_dir,
    )
    bundle = tune_and_save_thresholds(
        bundle,
        model_path,
        splits["validation"]["texts"],
        splits["test"]["texts"],
        dataset.other_texts,
        report_dir,
    )
    write_other_holdout_report(bundle, dataset.other_texts, dataset.other_paths, report_dir)
    write_inference_benchmark(bundle, model_path, splits["test"]["texts"], report_dir)
    print(f"Wrote evaluation reports to {report_dir}")


if __name__ == "__main__":
    main()
