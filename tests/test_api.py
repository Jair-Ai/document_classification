"""Behavioral tests for the document classification API.

Covers the health probe, the happy path, every documented error status
(400, 422, 500, 503), top_k handling, and the label contract. Runs
entirely against the self-contained fixture bundle from ``conftest.py``.
"""

from typing import Any, NoReturn

import pytest
from fastapi.testclient import TestClient

import app.main

VALID_TEXT = "The team won the championship final after the striker scored twice"

RESPONSE_FIELDS = {"message", "label", "raw_label", "confidence", "decision", "top_k"}
VALID_DECISIONS = {"auto_accept", "review_recommended", "manual_review", "fallback_other"}


class TestHealth:
    def test_health_ok_with_model_loaded(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}

    def test_health_ok_without_model(self, degraded_client: TestClient) -> None:
        response = degraded_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": False}


class TestClassifyDocument:
    def test_valid_document_returns_full_response(self, client: TestClient) -> None:
        response = client.post(
            "/classify_document", json={"document_text": VALID_TEXT, "top_k": 2}
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == RESPONSE_FIELDS
        assert body["message"] == "Classification successful"
        assert isinstance(body["label"], str)
        assert isinstance(body["raw_label"], str)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["decision"] in VALID_DECISIONS
        assert len(body["top_k"]) == 2
        for entry in body["top_k"]:
            assert set(entry) == {"label", "confidence"}
            assert 0.0 <= entry["confidence"] <= 1.0

    def test_top_k_defaults_to_three(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 200
        assert len(response.json()["top_k"]) == 3

    def test_top_k_clamped_to_trained_label_count(
        self, client: TestClient, trained_labels: list[str]
    ) -> None:
        response = client.post(
            "/classify_document", json={"document_text": VALID_TEXT, "top_k": 10}
        )

        assert response.status_code == 200
        assert len(response.json()["top_k"]) == len(trained_labels)

    def test_label_is_trained_or_other(
        self, client: TestClient, trained_labels: list[str]
    ) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 200
        body = response.json()
        assert body["label"] in {*trained_labels, "other"}
        assert body["raw_label"] in trained_labels

    def test_response_includes_request_id_header(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 200
        assert response.headers.get("x-request-id")


class TestValidationErrors:
    def test_missing_document_text_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"top_k": 3})

        assert response.status_code == 422

    def test_whitespace_only_text_returns_400(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": "   \n\t  "})

        assert response.status_code == 400
        assert response.json() == {"detail": "document_text must not be empty"}

    def test_top_k_zero_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/classify_document", json={"document_text": VALID_TEXT, "top_k": 0}
        )

        assert response.status_code == 422

    def test_top_k_above_maximum_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/classify_document", json={"document_text": VALID_TEXT, "top_k": 11}
        )

        assert response.status_code == 422

    def test_text_over_length_limit_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": "a" * 100_001})

        assert response.status_code == 422


class TestInferenceFailure:
    def test_unexpected_inference_error_returns_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_predict(*args: Any, **kwargs: Any) -> NoReturn:
            raise RuntimeError("simulated inference failure")

        monkeypatch.setattr(app.main, "predict_text", broken_predict)

        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 500
        assert response.json() == {"detail": "Unexpected classification error"}


class TestDegradedMode:
    def test_classify_without_model_returns_503(self, degraded_client: TestClient) -> None:
        response = degraded_client.post(
            "/classify_document", json={"document_text": VALID_TEXT}
        )

        assert response.status_code == 503
        assert response.json() == {"detail": "Model is unavailable"}
