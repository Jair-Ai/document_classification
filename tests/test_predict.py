"""Unit tests for the frozen prediction contract."""

from typing import Any

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import predict_batch, predict_text


@pytest.fixture()
def tiny_bundle() -> dict[str, Any]:
    """Train a small contract-compatible bundle for prediction tests."""
    examples = [
        ("stocks profits market investors revenue", "business"),
        ("company earnings shares merger bank", "business"),
        ("football match coach striker championship", "sport"),
        ("tennis player scored tournament final", "sport"),
        ("software chip computer security update", "technology"),
        ("robot cloud browser machine learning", "technology"),
    ]
    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(texts, labels)
    target_names = [str(label) for label in model.classes_]
    return {
        "model": model,
        "target_names": target_names,
        "model_type": "test_tfidf_logistic_regression",
        "trained_labels": target_names,
        "excluded_training_labels": ["other"],
        "label_aliases": {"technologie": "technology"},
        "confidence_thresholds": {
            "auto_accept": 0.95,
            "manual_review": 0.70,
            "other": 0.15,
        },
    }


def test_predict_text_returns_contract_fields(tiny_bundle: dict[str, Any]) -> None:
    """Prediction output contains the five public contract fields."""
    result = predict_text(
        "The company reported higher revenue and profits to investors.",
        tiny_bundle,
        top_k=2,
    )

    assert set(result) == {"label", "raw_label", "confidence", "decision", "top_k"}
    assert result["raw_label"] in tiny_bundle["target_names"]
    assert result["label"] in [*tiny_bundle["target_names"], "other"]
    assert 0 <= result["confidence"] <= 1


def test_predict_text_respects_top_k(tiny_bundle: dict[str, Any]) -> None:
    """Top-k is clamped to the requested length and sorted descending."""
    result = predict_text(
        "The tennis player won the championship match.",
        tiny_bundle,
        top_k=2,
    )

    assert len(result["top_k"]) == 2
    assert result["top_k"][0]["confidence"] >= result["top_k"][1]["confidence"]


def test_predict_text_clamps_top_k_to_available_labels(tiny_bundle: dict[str, Any]) -> None:
    """Requesting too many labels returns only available model labels."""
    result = predict_text(
        "The browser security update fixed software bugs.",
        tiny_bundle,
        top_k=20,
    )

    assert len(result["top_k"]) == len(tiny_bundle["target_names"])


def test_low_confidence_prediction_falls_back_to_other(tiny_bundle: dict[str, Any]) -> None:
    """Low confidence documents are routed to the fallback label."""
    tiny_bundle["confidence_thresholds"] = {
        **tiny_bundle["confidence_thresholds"],
        "other": 0.99,
    }

    result = predict_text(
        "A quiet garden festival invited neighbors to trade handwritten poems.",
        tiny_bundle,
        top_k=3,
    )

    assert result["confidence"] < tiny_bundle["confidence_thresholds"]["other"]
    assert result["label"] == "other"
    assert result["decision"] == "fallback_other"


def test_predict_batch_matches_single_prediction_contract(tiny_bundle: dict[str, Any]) -> None:
    """Batch predictions share the same routing contract as single predictions."""
    texts = [
        "The company reported higher revenue and profits to investors.",
        "The tennis player won the championship match.",
    ]

    batch_results = predict_batch(texts, tiny_bundle, top_k=2)
    single_results = [predict_text(text, tiny_bundle, top_k=2) for text in texts]

    assert len(batch_results) == 2
    assert batch_results == single_results
