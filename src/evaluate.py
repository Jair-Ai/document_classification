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
    ThresholdTradeoff,
    choose_other_threshold,
    confidence_range_report,
    pool_tradeoffs,
    threshold_tradeoff_table,
    threshold_tradeoffs,
    top_confidences,
)
from src.config import DEFAULT_THRESHOLDS, RANDOM_STATE, THRESHOLD_SEARCH, TRAINED_LABELS
from src.data_loader import load_dataset
from src.predict import predict_batch, predict_text
from src.train import build_model, stratified_split

#: Characters of source text kept in per-row report excerpts.
EXCERPT_CHARS = 500


def raw_predictions(bundle: dict[str, Any], texts: list[str]) -> list[str]:
    """Return raw model labels, before threshold routing."""
    return [str(label) for label in bundle["model"].predict(texts)]


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Collapse whitespace and truncate to a single-line report excerpt."""
    return " ".join(text.split())[:limit]


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


def write_misclassified_examples(
    bundle: dict[str, Any],
    test_texts: list[str],
    test_labels: list[str],
    test_paths: list[str],
    report_dir: Path,
) -> None:
    """Write every test-set document whose argmax label is wrong.

    Rows are keyed on the raw model prediction (``raw_label != actual``),
    so a misclassified document also shows its post-routing ``final_label``
    and ``decision`` — a row can be both misclassified and routed to
    ``other``. ``top_k`` is JSON-encoded so the notebook can parse the
    full probability vector back out.
    """
    predictions = predict_batch(test_texts, bundle, top_k=3)
    rows = [
        {
            "path": path,
            "actual_label": actual,
            "raw_label": pred["raw_label"],
            "final_label": pred["label"],
            "confidence": round(pred["confidence"], 4),
            "decision": pred["decision"],
            "top_k": json.dumps(pred["top_k"]),
            "text_excerpt": _excerpt(text),
        }
        for path, text, actual, pred in zip(
            test_paths, test_texts, test_labels, predictions, strict=True
        )
        if pred["raw_label"] != actual
    ]
    pd.DataFrame(
        rows,
        columns=[
            "path",
            "actual_label",
            "raw_label",
            "final_label",
            "confidence",
            "decision",
            "top_k",
            "text_excerpt",
        ],
    ).to_csv(report_dir / "misclassified_examples.csv", index=False)


def write_other_holdout_predictions(
    bundle: dict[str, Any],
    other_texts: list[str],
    other_paths: list[str],
    report_dir: Path,
) -> None:
    """Write the per-document routing decision for the OOD holdout.

    Unlike :func:`write_other_holdout_report` (which summarizes routing),
    this emits one row per holdout document so the error-analysis notebook
    can inspect where each ``other`` file landed.
    """
    predictions = predict_batch(other_texts, bundle, top_k=3)
    rows = [
        {
            "path": path,
            "final_label": pred["label"],
            "raw_label": pred["raw_label"],
            "confidence": round(pred["confidence"], 4),
            "decision": pred["decision"],
            "top_k": json.dumps(pred["top_k"]),
            "text_excerpt": _excerpt(text),
        }
        for path, text, pred in zip(other_paths, other_texts, predictions, strict=True)
    ]
    pd.DataFrame(
        rows,
        columns=[
            "path",
            "final_label",
            "raw_label",
            "confidence",
            "decision",
            "top_k",
            "text_excerpt",
        ],
    ).to_csv(report_dir / "other_holdout_predictions.csv", index=False)


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


def leave_one_class_out_ood(
    known_texts: list[str],
    known_labels: list[str],
    known_paths: list[str],
    model_name: str,
    val_size: float,
    test_size: float,
    random_state: int,
    thresholds: list[float] | None = None,
) -> list[ThresholdTradeoff]:
    """Estimate OOD catch rate with a leave-one-class-out probe.

    The shipped six-file ``other`` holdout is too small to bound the
    fallback false-positive/true-positive trade-off. This diagnostic
    synthesizes a much larger OOD set *without new labels*: each known
    class is excluded from training in turn and its documents are treated
    as pseudo-OOD, while a held-in validation slice of the remaining
    classes measures the known-label misroute side. Per-class folds are
    pooled into a single threshold curve, so the shipped ``other``
    threshold can be read against hundreds of OOD-like documents instead
    of six. It re-trains the model per fold and never touches the shipped
    bundle — it is a diagnostic, not a second threshold selector.
    """
    fold_rows: list[ThresholdTradeoff] = []
    for held_out in sorted(set(known_labels)):
        rest = [
            (text, label, path)
            for text, label, path in zip(known_texts, known_labels, known_paths, strict=True)
            if label != held_out
        ]
        ood_texts = [
            text
            for text, label in zip(known_texts, known_labels, strict=True)
            if label == held_out
        ]
        splits = stratified_split(
            [text for text, _, _ in rest],
            [label for _, label, _ in rest],
            [path for _, _, path in rest],
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )
        model = build_model(model_name)
        model.fit(splits["train"]["texts"], splits["train"]["labels"])
        fold_rows.extend(
            threshold_tradeoffs(
                {"model": model},
                splits["validation"]["texts"],
                ood_texts,
                thresholds,
            )
        )
    return pool_tradeoffs(fold_rows)


def write_loco_ood_curve(
    known_texts: list[str],
    known_labels: list[str],
    known_paths: list[str],
    model_name: str,
    val_size: float,
    test_size: float,
    random_state: int,
    report_dir: Path,
) -> list[ThresholdTradeoff]:
    """Run the leave-one-class-out OOD probe and write its pooled curve."""
    rows = leave_one_class_out_ood(
        known_texts,
        known_labels,
        known_paths,
        model_name,
        val_size,
        test_size,
        random_state,
    )
    # Rename the holdout columns: here they count pooled pseudo-OOD docs
    # (held-out known classes), not the six-file ``other`` holdout.
    threshold_tradeoff_table(rows).rename(
        columns={
            "other_holdout_docs": "pseudo_ood_docs",
            "other_holdout_caught": "pseudo_ood_caught",
            "other_holdout_caught_pct": "pseudo_ood_caught_pct",
        }
    ).to_csv(report_dir / "ood_loco_curve.csv", index=False)
    return rows


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
    loco_rows: list[ThresholdTradeoff] | None = None,
) -> dict[str, Any]:
    """Choose the fallback threshold on validation and persist it."""
    validation_rows = threshold_tradeoffs(bundle, val_texts, other_texts)
    choice = choose_other_threshold(validation_rows)

    thresholds = dict(bundle.get("confidence_thresholds", {}))
    thresholds.update(
        {
            "auto_accept": DEFAULT_THRESHOLDS["auto_accept"],
            "manual_review": DEFAULT_THRESHOLDS["manual_review"],
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
        "threshold_search": {
            "other_min": THRESHOLD_SEARCH.other_min,
            "other_max": THRESHOLD_SEARCH.other_max,
            "other_step": THRESHOLD_SEARCH.other_step,
            "max_known_misroute_pct": THRESHOLD_SEARCH.max_known_misroute_pct,
        },
        "selection_basis": "validation known-label misroute guardrail plus anecdotal OOD holdout",
        "reason": choice.reason,
        "other_holdout_note": (
            "The OOD side has only six files, so it is useful as a smoke "
            "check but too small for a statistically stable threshold."
        ),
    }
    if loco_rows is not None:
        pooled = next((row for row in loco_rows if row.threshold == choice.other_threshold), None)
        if pooled is not None:
            selection["ood_loco_validation"] = {
                "method": "leave_one_class_out",
                "pseudo_ood_docs": pooled.other_holdout_docs,
                "pseudo_ood_caught_pct": pooled.other_holdout_caught_pct,
                "note": (
                    "Larger synthetic OOD probe at the chosen threshold; each "
                    "known class is held out of training in turn and treated as "
                    "OOD. See ood_loco_curve.csv for the full curve."
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
    parser.add_argument(
        "--data-dir",
        default="data/trellis_assessment_ds",
        help="Path to the assessment dataset (default: data/trellis_assessment_ds)",
    )
    parser.add_argument("--model-path", default="models/document_classifier.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--loco",
        action="store_true",
        help=(
            "Also run the leave-one-class-out OOD diagnostic. Off by default: "
            "it re-trains the model once per known class, which the core reports "
            "do not require. When set, writes reports/ood_loco_curve.csv and adds "
            "an ood_loco_validation block to threshold_selection.json."
        ),
    )
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
    # Opt-in: the diagnostic re-trains per class, so the default run stays
    # load-only. Pass --loco to also produce the leave-one-class-out curve.
    loco_rows = None
    if args.loco:
        loco_rows = write_loco_ood_curve(
            dataset.known_texts,
            dataset.known_labels,
            dataset.known_paths,
            bundle["model_type"],
            args.val_size,
            args.test_size,
            args.random_state,
            report_dir,
        )
    bundle = tune_and_save_thresholds(
        bundle,
        model_path,
        splits["validation"]["texts"],
        splits["test"]["texts"],
        dataset.other_texts,
        report_dir,
        loco_rows=loco_rows,
    )
    # Run after tuning so final_label/decision reflect the shipped policy.
    write_misclassified_examples(
        bundle,
        splits["test"]["texts"],
        splits["test"]["labels"],
        splits["test"]["paths"],
        report_dir,
    )
    write_other_holdout_report(bundle, dataset.other_texts, dataset.other_paths, report_dir)
    write_other_holdout_predictions(bundle, dataset.other_texts, dataset.other_paths, report_dir)
    write_inference_benchmark(bundle, model_path, splits["test"]["texts"], report_dir)
    print(f"Wrote evaluation reports to {report_dir}")


if __name__ == "__main__":
    main()
