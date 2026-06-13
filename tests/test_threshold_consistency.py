"""Guards against confidence-threshold drift across the repo.

The `other` fallback threshold is data-tuned by ``src/evaluate.py`` and
baked into the shipped model bundle. Three other places restate that
policy for humans and clients: ``config/model.toml`` (the declared seed),
``reports/threshold_selection.json`` (the recorded tuning output), and
``docs/readme_api.md`` (the client-facing contract). If any of them
disagree, a client or teammate can be misled about how the service
routes by confidence. These tests fail loudly when they drift apart.
"""

import json
import math
import tomllib
from pathlib import Path

import joblib
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BUNDLE_PATH = _ROOT / "models" / "document_classifier.joblib"
_SELECTION_PATH = _ROOT / "reports" / "threshold_selection.json"
_CONFIG_PATH = _ROOT / "config" / "model.toml"
_API_DOC_PATH = _ROOT / "docs" / "readme_api.md"

_THRESHOLD_KEYS = ("auto_accept", "manual_review", "other")

requires_bundle = pytest.mark.skipif(
    not _BUNDLE_PATH.exists(),
    reason="model bundle not present (run training/evaluation first)",
)


def _format_threshold(value: float) -> str:
    """Render a threshold the way it appears in prose (e.g. 0.15)."""
    return f"{value:g}"


@requires_bundle
def test_shipped_bundle_matches_selection_report() -> None:
    """The artifact we serve and the recorded tuning output must agree."""
    bundle = joblib.load(_BUNDLE_PATH)
    shipped = bundle["confidence_thresholds"]
    recorded = json.loads(_SELECTION_PATH.read_text())["confidence_thresholds"]

    for key in _THRESHOLD_KEYS:
        assert math.isclose(shipped[key], recorded[key], abs_tol=1e-9), (
            f"{key}: bundle={shipped[key]} != threshold_selection.json={recorded[key]}"
        )


@requires_bundle
def test_declared_config_matches_shipped_bundle() -> None:
    """config/model.toml is the declared policy; it must match what ships.

    A retrain that tunes ``other`` to a new value should be accompanied by
    an update to config/model.toml, so this test forces that discipline.
    """
    bundle = joblib.load(_BUNDLE_PATH)
    shipped = bundle["confidence_thresholds"]
    declared = tomllib.loads(_CONFIG_PATH.read_text())["thresholds"]

    for key in _THRESHOLD_KEYS:
        assert math.isclose(declared[key], shipped[key], abs_tol=1e-9), (
            f"{key}: config/model.toml={declared[key]} != shipped bundle={shipped[key]}"
        )


@requires_bundle
def test_api_doc_states_shipped_fallback_threshold() -> None:
    """The client-facing doc must quote the actual shipped fallback value."""
    bundle = joblib.load(_BUNDLE_PATH)
    other = bundle["confidence_thresholds"]["other"]
    api_doc = _API_DOC_PATH.read_text()

    assert f"below {_format_threshold(other)}" in api_doc, (
        f"docs/readme_api.md does not state the shipped fallback threshold "
        f"'below {_format_threshold(other)}'"
    )
