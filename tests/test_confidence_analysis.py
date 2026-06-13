"""Unit tests for threshold search defaults and selection helpers."""

from src.confidence_analysis import choose_other_threshold, threshold_tradeoffs
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
