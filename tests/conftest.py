"""Fixtures for the API test suite.

A tiny TF-IDF + logistic regression pipeline is trained on hardcoded
sentences and serialized in the exact bundle layout the application
expects. This keeps the API tests self-contained and fast: they verify
the service honors the bundle contract without depending on the full
training pipeline or the production artifact.

All fixtures use standard injection (no autouse, no global env
mutation) so they cannot affect other test modules.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Make the project root importable regardless of how pytest is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.main import app  # noqa: E402

_TRAINING_SENTENCES: list[tuple[str, str]] = [
    ("The central bank raised interest rates to curb inflation", "business"),
    ("Quarterly profits surged as the company beat revenue forecasts", "business"),
    ("Shares fell sharply after the merger talks collapsed", "business"),
    ("The startup secured millions in venture capital funding", "business"),
    ("Oil prices climbed amid supply concerns in global markets", "business"),
    ("Investors weighed earnings reports ahead of the trading session", "business"),
    ("The retailer announced layoffs to cut operating costs", "business"),
    ("The striker scored twice as the team won the championship final", "sports"),
    ("The coach praised his players after a hard fought victory", "sports"),
    ("She broke the world record in the hundred meter sprint", "sports"),
    ("The tennis star advanced to the semifinals in straight sets", "sports"),
    ("Fans celebrated as the club lifted the league trophy", "sports"),
    ("The quarterback threw three touchdowns in the season opener", "sports"),
    ("Olympic athletes began training for the upcoming games", "sports"),
    ("The new smartphone features a faster chip and better camera", "technology"),
    ("Researchers unveiled software that detects bugs automatically", "technology"),
    ("The company released an update patching critical security flaws", "technology"),
    ("Engineers built a robot capable of navigating rough terrain", "technology"),
    ("The browser update improves privacy controls for users", "technology"),
    ("Cloud computing adoption keeps growing among enterprises", "technology"),
    ("Developers praised the open source machine learning library", "technology"),
]


@pytest.fixture(scope="session")
def fixture_bundle() -> dict[str, Any]:
    """Train the small pipeline and wrap it in the bundle layout."""
    texts = [text for text, _ in _TRAINING_SENTENCES]
    labels = [label for _, label in _TRAINING_SENTENCES]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(texts, labels)

    target_names = [str(name) for name in pipeline.classes_]
    return {
        "model": pipeline,
        "target_names": target_names,
        "model_type": "tfidf_logistic_regression",
        "trained_labels": target_names,
        "excluded_training_labels": ["other"],
        "label_aliases": {"technologie": "technology"},
        "confidence_thresholds": {
            "auto_accept": 0.9,
            "manual_review": 0.7,
            "other": 0.55,
        },
    }


@pytest.fixture(scope="session")
def fixture_bundle_path(
    fixture_bundle: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Serialize the fixture bundle to a temporary joblib artifact."""
    path = tmp_path_factory.mktemp("model") / "fixture_classifier.joblib"
    joblib.dump(fixture_bundle, path)
    return path


@pytest.fixture(scope="session")
def trained_labels(fixture_bundle: dict[str, Any]) -> list[str]:
    """Labels the fixture model was trained on."""
    return list(fixture_bundle["trained_labels"])


@pytest.fixture()
def client(fixture_bundle_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient against an app that loaded the fixture bundle."""
    monkeypatch.setenv("MODEL_PATH", str(fixture_bundle_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def degraded_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient against an app whose model artifact does not exist."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    with TestClient(app) as test_client:
        yield test_client
