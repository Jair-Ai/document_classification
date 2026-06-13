"""Unit tests for threshold search defaults and selection helpers."""

import numpy as np

from src.confidence_analysis import choose_other_threshold, ood_score_auc, threshold_tradeoffs
from src.config import THRESHOLD_SEARCH


def test_threshold_tradeoffs_uses_configured_search_space() -> None:
    class DummyModel:
        def predict_proba(self, texts: list[str]) -> list[list[float]]:
            if len(texts) == 3:
                return [[0.95], [0.72], [0.18]]
            return [[0.08], [0.61]]

    bundle = {"model": DummyModel()}

    rows = threshold_tradeoffs(bundle, ["a", "b", "c"], ["x", "y"])

    assert rows[0].threshold == THRESHOLD_SEARCH.other_min
    assert rows[-1].threshold == THRESHOLD_SEARCH.other_max
    assert (
        len(rows)
        == int(
            round(
                (THRESHOLD_SEARCH.other_max - THRESHOLD_SEARCH.other_min)
                / THRESHOLD_SEARCH.other_step
            )
        )
        + 1
    )


def test_choose_other_threshold_uses_configured_guardrail() -> None:
    rows = threshold_tradeoffs(
        {
            "model": type(
                "DummyModel",
                (),
                {
                    "predict_proba": lambda self, texts: (
                        [[0.96], [0.80], [0.28]] if len(texts) == 3 else [[0.03], [0.19]]
                    )
                },
            )()
        },
        ["a", "b", "c"],
        ["x", "y"],
        thresholds=[0.05, 0.10, 0.20],
    )

    choice = choose_other_threshold(rows)

    assert choice.other_threshold == 0.2
    assert f"{THRESHOLD_SEARCH.max_known_misroute_pct:.0%}" in choice.reason


def test_ood_score_auc_perfectly_separable() -> None:
    """High-confidence known vs low-confidence OOD is perfectly separable."""
    known = np.array([0.90, 0.95, 0.99])
    ood = np.array([0.05, 0.10, 0.20])

    result = ood_score_auc(known, ood)

    assert result.auroc == 1.0
    assert result.aupr == 1.0
    assert result.known_docs == 3
    assert result.ood_docs == 3


def test_ood_score_auc_uses_ood_as_positive_class() -> None:
    """The score is ``1 - max_softmax``; OOD (low confidence) is positive.

    If the sign were inverted, this clearly separable case would score
    near 0 instead of 1, so this pins the score direction.
    """
    known = np.array([0.80, 0.85, 0.92, 0.88])
    ood = np.array([0.10, 0.12, 0.15])

    result = ood_score_auc(known, ood)

    assert result.auroc > 0.9


def test_ood_score_auc_chance_for_indistinguishable_scores() -> None:
    """Identical confidences on both sides give chance-level AUROC."""
    known = np.full(5, 0.5)
    ood = np.full(5, 0.5)

    result = ood_score_auc(known, ood)

    assert result.auroc == 0.5
