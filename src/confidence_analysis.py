"""Confidence analysis helpers for threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdChoice:
    """Selected fallback threshold plus the reason it was chosen."""

    other_threshold: float
    reason: str


@dataclass(frozen=True)
class ThresholdTradeoff:
    """Known-label recall and OOD catch metrics for one fallback threshold."""

    threshold: float
    known_docs: int
    known_misrouted_to_other: int
    known_misrouted_pct: float
    other_holdout_docs: int
    other_holdout_caught: int
    other_holdout_caught_pct: float | None


def top_confidences(bundle: dict[str, Any], texts: list[str]) -> np.ndarray:
    """Return the maximum predicted probability for each text."""
    probabilities = bundle["model"].predict_proba(texts)
    return np.asarray(probabilities).max(axis=1)


def confidence_range_report(
    confidences: np.ndarray,
    correct: np.ndarray,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Summarize accuracy by confidence bucket."""
    if bins is None:
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rows = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if upper == bins[-1]:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        total = int(mask.sum())
        correct_count = int(correct[mask].sum()) if total else 0
        rows.append(
            {
                "confidence_range": f"[{lower:.1f}, {upper:.1f}{']' if upper == bins[-1] else ')'}",
                "total": total,
                "correct": correct_count,
                "accuracy": round(correct_count / total, 4) if total else None,
            }
        )
    return pd.DataFrame(rows)


def threshold_tradeoffs(
    bundle: dict[str, Any],
    known_texts: list[str],
    other_texts: list[str],
    thresholds: list[float] | None = None,
) -> list[ThresholdTradeoff]:
    """Compare fallback thresholds on known recall and OOD catch rate.

    A known document is "misrouted" when its top probability falls below
    the candidate ``other`` threshold and would therefore be assigned the
    fallback label by ``predict_text``. An OOD holdout document is
    "caught" by the same rule.
    """
    if thresholds is None:
        thresholds = [round(value, 2) for value in np.arange(0.10, 0.66, 0.05)]

    known_conf = top_confidences(bundle, known_texts)
    other_conf = top_confidences(bundle, other_texts) if other_texts else np.array([])

    rows: list[ThresholdTradeoff] = []
    for threshold in thresholds:
        known_misrouted = int((known_conf < threshold).sum())
        other_caught = int((other_conf < threshold).sum())
        rows.append(
            ThresholdTradeoff(
                threshold=float(threshold),
                known_docs=int(len(known_conf)),
                known_misrouted_to_other=known_misrouted,
                known_misrouted_pct=round(known_misrouted / len(known_conf), 4),
                other_holdout_docs=int(len(other_conf)),
                other_holdout_caught=other_caught,
                other_holdout_caught_pct=(
                    round(other_caught / len(other_conf), 4) if len(other_conf) else None
                ),
            )
        )
    return rows


def threshold_tradeoff_table(rows: list[ThresholdTradeoff]) -> pd.DataFrame:
    """Convert typed threshold records to a report-friendly DataFrame."""
    return pd.DataFrame([row.__dict__ for row in rows])


def choose_other_threshold(
    rows: list[ThresholdTradeoff],
    max_known_misroute_pct: float = 0.05,
) -> ThresholdChoice:
    """Pick the highest OOD catch rate under a known-recall guardrail."""
    eligible = [row for row in rows if row.known_misrouted_pct <= max_known_misroute_pct]
    if not eligible:
        best = min(
            rows,
            key=lambda row: (row.known_misrouted_pct, -row.other_holdout_caught),
        )
        return ThresholdChoice(
            other_threshold=best.threshold,
            reason=(
                "No candidate met the known-recall guardrail; chose the lowest known misroute rate."
            ),
        )

    best = max(
        eligible,
        key=lambda row: (row.other_holdout_caught, row.threshold),
    )
    return ThresholdChoice(
        other_threshold=best.threshold,
        reason=(
            f"Chose the highest OOD catch count with known misroutes <= "
            f"{max_known_misroute_pct:.0%} on validation."
        ),
    )
