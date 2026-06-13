"""Behavioral tests for the document classification API.

Covers the health probe, the happy path, every documented error status
(400, 422, 500, 503), top_k handling, and the label contract. Runs
entirely against the self-contained fixture bundle from ``conftest.py``.
"""

from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
from fastapi.testclient import TestClient

import app.main
from app.config import settings

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
        response = client.post("/classify_document", json={"document_text": VALID_TEXT, "top_k": 2})

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

    def test_label_is_trained_or_other(self, client: TestClient, trained_labels: list[str]) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 200
        body = response.json()
        assert body["label"] in {*trained_labels, "other"}
        assert body["raw_label"] in trained_labels

    def test_response_includes_request_id_header(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 200
        assert response.headers.get("x-request-id")


class TestClassifyDocumentsBatch:
    def test_valid_batch_returns_result_per_document(self, client: TestClient) -> None:
        response = client.post(
            "/classify_documents",
            json={
                "documents": [
                    {"id": "doc-1", "document_text": VALID_TEXT},
                    {
                        "id": "doc-2",
                        "document_text": "The company reported profits and market revenue",
                    },
                ],
                "top_k": 2,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Batch classification successful"
        assert body["total"] == 2
        assert len(body["results"]) == 2
        assert body["results"][0]["document_id"] == "doc-1"
        assert len(body["results"][0]["top_k"]) == 2
        assert set(body["results"][0]) == {
            "document_id",
            "label",
            "raw_label",
            "confidence",
            "decision",
            "top_k",
        }

    def test_batch_rejects_empty_documents_list(self, client: TestClient) -> None:
        response = client.post("/classify_documents", json={"documents": []})

        assert response.status_code == 422

    def test_batch_rejects_too_many_documents(self, client: TestClient) -> None:
        response = client.post(
            "/classify_documents",
            json={
                "documents": [
                    {"id": f"doc-{index}", "document_text": VALID_TEXT} for index in range(101)
                ]
            },
        )

        assert response.status_code == 422

    def test_batch_reports_whitespace_document_index(self, client: TestClient) -> None:
        response = client.post(
            "/classify_documents",
            json={
                "documents": [
                    {"id": "good", "document_text": VALID_TEXT},
                    {"id": "blank", "document_text": "   "},
                ]
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "detail": {
                "message": "document_text must not be empty",
                "index": 1,
                "document_id": "blank",
            }
        }

    def test_batch_without_model_returns_503(self, degraded_client: TestClient) -> None:
        response = degraded_client.post(
            "/classify_documents",
            json={"documents": [{"id": "doc-1", "document_text": VALID_TEXT}]},
        )

        assert response.status_code == 503
        assert response.json() == {"detail": "Model is unavailable"}

    def test_batch_requires_api_key_when_enabled(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            app.main,
            "settings",
            TestApiKeyAuth._auth_settings(enabled=True),
        )

        response = client.post(
            "/classify_documents",
            json={"documents": [{"id": "doc-1", "document_text": VALID_TEXT}]},
        )

        assert response.status_code == 401


class TestValidationErrors:
    def test_missing_document_text_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"top_k": 3})

        assert response.status_code == 422

    def test_whitespace_only_text_returns_400(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": "   \n\t  "})

        assert response.status_code == 400
        assert response.json() == {"detail": "document_text must not be empty"}

    def test_top_k_zero_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT, "top_k": 0})

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
        response = degraded_client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 503
        assert response.json() == {"detail": "Model is unavailable"}


class TestApiKeyAuth:
    @staticmethod
    def _auth_settings(
        *,
        enabled: bool,
        key: str = "secret-test-key",
        header: str = "X-API-Key",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            api=settings.api,
            security=SimpleNamespace(
                api_key_enabled=enabled,
                api_key=key,
                api_key_header=header,
            ),
        )

    def test_api_key_disabled_by_default(self, client: TestClient) -> None:
        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 200

    def test_enabled_api_key_rejects_missing_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app.main, "settings", self._auth_settings(enabled=True))

        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid or missing API key"}
        assert response.headers["www-authenticate"] == "ApiKey"

    def test_enabled_api_key_rejects_invalid_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app.main, "settings", self._auth_settings(enabled=True))

        response = client.post(
            "/classify_document",
            json={"document_text": VALID_TEXT},
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid or missing API key"}

    def test_enabled_api_key_accepts_valid_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app.main, "settings", self._auth_settings(enabled=True))

        response = client.post(
            "/classify_document",
            json={"document_text": VALID_TEXT},
            headers={"X-API-Key": "secret-test-key"},
        )

        assert response.status_code == 200

    def test_health_is_exempt_from_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app.main, "settings", self._auth_settings(enabled=True))

        response = client.get("/health")

        assert response.status_code == 200

    def test_enabled_api_key_without_secret_returns_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app.main, "settings", self._auth_settings(enabled=True, key=""))

        response = client.post("/classify_document", json={"document_text": VALID_TEXT})

        assert response.status_code == 500
        assert response.json() == {"detail": "API key authentication is misconfigured"}


class TestClassifyFile:
    @staticmethod
    def _upload(name: str, content: bytes) -> dict[str, tuple[str, bytes, str]]:
        return {"file": (name, content, "text/plain")}

    def test_txt_upload_returns_full_response(self, client: TestClient) -> None:
        response = client.post(
            "/classify-file", files=self._upload("doc.txt", VALID_TEXT.encode("utf-8"))
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == RESPONSE_FIELDS
        assert body["message"] == "Classification successful"
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["decision"] in VALID_DECISIONS
        assert len(body["top_k"]) == 3

    def test_top_k_query_param_is_respected(self, client: TestClient) -> None:
        response = client.post(
            "/classify-file",
            params={"top_k": 2},
            files=self._upload("doc.txt", VALID_TEXT.encode("utf-8")),
        )

        assert response.status_code == 200
        assert len(response.json()["top_k"]) == 2

    def test_non_txt_extension_returns_400(self, client: TestClient) -> None:
        response = client.post("/classify-file", files=self._upload("doc.pdf", b"not a text file"))

        assert response.status_code == 400
        assert response.json() == {"detail": "Only .txt files are supported"}

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify-file")

        assert response.status_code == 422

    def test_whitespace_only_file_returns_400(self, client: TestClient) -> None:
        response = client.post("/classify-file", files=self._upload("doc.txt", b"  \n\t "))

        assert response.status_code == 400
        assert response.json() == {"detail": "document_text must not be empty"}

    def test_invalid_utf8_bytes_are_ignored(self, client: TestClient) -> None:
        content = VALID_TEXT.encode("utf-8") + b"\xff\xfe\xff"

        response = client.post("/classify-file", files=self._upload("doc.txt", content))

        assert response.status_code == 200
        assert response.json()["message"] == "Classification successful"

    def test_oversized_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/classify-file", files=self._upload("doc.txt", b"a" * 100_001))

        assert response.status_code == 422

    def test_file_upload_without_model_returns_503(self, degraded_client: TestClient) -> None:
        response = degraded_client.post(
            "/classify-file", files=self._upload("doc.txt", VALID_TEXT.encode("utf-8"))
        )

        assert response.status_code == 503
        assert response.json() == {"detail": "Model is unavailable"}
