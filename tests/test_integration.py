"""End-to-end integration tests against the real artifact and dataset.

Unlike the fast suite in ``test_api.py`` (which runs against the tiny
fixture bundle from ``conftest.py``), these tests close the seam between
the trained ``models/document_classifier.joblib`` artifact and the
running service: they load the real bundle and push real dataset
documents through the HTTP layer, asserting the full response contract.

They are slow and depend on the large artifact/dataset, which may not be
present locally or in CI, so they are marked ``integration`` and excluded
from the default ``pytest`` run (see ``addopts`` in ``pyproject.toml``).
Run them explicitly with ``pytest -m integration``.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = _PROJECT_ROOT / "models" / "document_classifier.joblib"
DATA = Path(os.environ.get("DATA_DIR", _PROJECT_ROOT / "data" / "trellis_assessment_ds"))

# Contract the service promises on a successful classification. Kept in
# sync with ``ClassificationResponse`` in ``app.schemas``.
RESPONSE_FIELDS = {"message", "label", "raw_label", "confidence", "decision", "top_k"}
VALID_DECISIONS = {"auto_accept", "review_recommended", "manual_review", "fallback_other"}

# A few trained labels whose dataset folder name differs from the label
# the model emits (the dataset ships a misspelled ``technologie`` folder).
_LABEL_TO_DIR = {"technology": "technologie"}

requires_artifact = pytest.mark.skipif(not ARTIFACT.exists(), reason="real artifact not present")
requires_dataset = pytest.mark.skipif(not DATA.exists(), reason="dataset not present")


@pytest.fixture()
def real_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient against an app that loaded the real artifact."""
    monkeypatch.setenv("MODEL_PATH", str(ARTIFACT))
    with TestClient(app) as client:
        yield client


def _first_doc(label_dir: str) -> str:
    """Return the text of the first ``.txt`` document under a dataset folder."""
    sample = next((DATA / label_dir).glob("*.txt"))
    return sample.read_text(encoding="utf-8", errors="ignore")


def _trained_labels() -> list[str]:
    """The labels the real model was trained on, read from the artifact."""
    import joblib

    bundle = joblib.load(ARTIFACT)
    return list(bundle["trained_labels"])


def _assert_valid_contract(body: dict[str, object]) -> None:
    """Assert a 200 body satisfies the full classification contract."""
    assert set(body) == RESPONSE_FIELDS
    assert isinstance(body["label"], str) and body["label"]
    assert isinstance(body["raw_label"], str) and body["raw_label"]
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["decision"] in VALID_DECISIONS
    assert isinstance(body["top_k"], list) and body["top_k"]


@requires_artifact
@requires_dataset
def test_health_reports_real_model_loaded(real_client: TestClient) -> None:
    response = real_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


@requires_artifact
@requires_dataset
@pytest.mark.parametrize("label", _trained_labels() if ARTIFACT.exists() else [])
def test_real_dataset_document_classifies(real_client: TestClient, label: str) -> None:
    """One real document per trained class flows through the HTTP layer."""
    document_text = _first_doc(_LABEL_TO_DIR.get(label, label))

    response = real_client.post("/classify_document", json={"document_text": document_text})

    assert response.status_code == 200
    body = response.json()
    _assert_valid_contract(body)
    # Routing may fall back to ``other`` on low confidence, so we assert the
    # contract rather than the exact label (model performance is not tested).
    assert body["label"] in {label, "other"}


@requires_artifact
@requires_dataset
def test_other_holdout_document_classifies(real_client: TestClient) -> None:
    """An unseen ``other`` document is served without breaking the contract."""
    document_text = _first_doc("other")

    response = real_client.post("/classify_document", json={"document_text": document_text})

    assert response.status_code == 200
    _assert_valid_contract(response.json())
