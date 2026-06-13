"""Tests for the leave-one-class-out OOD probe in ``src.evaluate``.

The probe synthesizes a larger out-of-distribution set than the six-file
``other`` holdout by holding each known class out of training in turn and
treating it as pseudo-OOD. These tests pin the pooled-curve schema and the
core pooling arithmetic so the artifact stays reproducible and the docs can
quote it.
"""

from pathlib import Path

import pandas as pd

from src.confidence_analysis import ThresholdTradeoff, pool_tradeoffs
from src.evaluate import write_loco_ood_curve

_LOCO_COLUMNS = [
    "threshold",
    "known_docs",
    "known_misrouted_to_other",
    "known_misrouted_pct",
    "pseudo_ood_docs",
    "pseudo_ood_caught",
    "pseudo_ood_caught_pct",
]

# Three lexically distinct classes so the tiny pipeline trains cleanly and
# every class can be held out as pseudo-OOD in turn.
_CLASS_TEMPLATES = {
    "business": "quarterly profits revenue markets investors earnings shares trading {n}",
    "sport": "striker scored championship final coach players victory league {n}",
    "technology": "smartphone chip software security update cloud computing developers {n}",
}


def _synthetic_known() -> tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    paths: list[str] = []
    for label, template in _CLASS_TEMPLATES.items():
        for index in range(12):
            texts.append(template.format(n=index))
            labels.append(label)
            paths.append(f"{label}_{index}.txt")
    return texts, labels, paths


def test_pool_tradeoffs_sums_folds() -> None:
    """Two folds at the same threshold pool into summed counts and rates."""
    fold_a = ThresholdTradeoff(
        threshold=0.15,
        known_docs=100,
        known_misrouted_to_other=2,
        known_misrouted_pct=0.02,
        other_holdout_docs=50,
        other_holdout_caught=10,
        other_holdout_caught_pct=0.2,
    )
    fold_b = ThresholdTradeoff(
        threshold=0.15,
        known_docs=100,
        known_misrouted_to_other=4,
        known_misrouted_pct=0.04,
        other_holdout_docs=50,
        other_holdout_caught=30,
        other_holdout_caught_pct=0.6,
    )

    pooled = pool_tradeoffs([fold_a, fold_b])

    assert len(pooled) == 1
    row = pooled[0]
    assert row.known_docs == 200
    assert row.known_misrouted_to_other == 6
    assert row.known_misrouted_pct == 0.03
    assert row.other_holdout_docs == 100
    assert row.other_holdout_caught == 40
    assert row.other_holdout_caught_pct == 0.4


def test_loco_ood_curve_schema_and_pooling(tmp_path: Path) -> None:
    """Curve has the frozen schema, pools every held-out doc, and rises."""
    texts, labels, paths = _synthetic_known()

    write_loco_ood_curve(
        texts,
        labels,
        paths,
        model_name="logistic_regression",
        val_size=0.25,
        test_size=0.25,
        random_state=42,
        report_dir=tmp_path,
    )

    df = pd.read_csv(tmp_path / "ood_loco_curve.csv")
    assert list(df.columns) == _LOCO_COLUMNS
    # Every document of every class is held out as pseudo-OOD exactly once.
    assert (df["pseudo_ood_docs"] == len(texts)).all()
    # Catch rate is monotonically non-decreasing as the threshold rises.
    assert df["pseudo_ood_caught_pct"].is_monotonic_increasing
    assert df["pseudo_ood_caught_pct"].between(0.0, 1.0).all()
