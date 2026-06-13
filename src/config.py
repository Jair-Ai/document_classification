"""Shared constants for data loading, training, and evaluation.

Centralizes label definitions and model-level defaults so the loader,
training script, prediction path, and evaluation code never disagree
about what the model is trained on or how confidence routing starts.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib

# The ten labels the classifier is trained on, sorted alphabetically.
# This order must match the column order of the model's predict_proba
# output (sklearn sorts classes_ lexicographically for string labels).
TRAINED_LABELS: list[str] = [
    "business",
    "entertainment",
    "food",
    "graphics",
    "historical",
    "medical",
    "politics",
    "space",
    "sport",
    "technology",
]

# Fallback label for low-confidence / out-of-distribution documents.
# Never trained on; assigned only by threshold routing in src.predict.
FALLBACK_LABEL: str = "other"

# Folder names on disk that must be normalized to a canonical label.
# The dataset ships with a "technologie" folder which is a typo for
# "technology" — folder names are mapped through this table.
LABEL_ALIASES: dict[str, str] = {"technologie": "technology"}

# Folders excluded from the training pool. "other" is kept aside as an
# out-of-distribution holdout used only after training.
EXCLUDED_TRAINING_LABELS: list[str] = [FALLBACK_LABEL]

# Default seed for every stochastic step (splits, models).
RANDOM_STATE: int = 42

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_FILE = _PROJECT_ROOT / "config" / "model.toml"


@dataclass(frozen=True)
class ThresholdSearchConfig:
    """Search-space settings for tuning the fallback threshold."""

    other_min: float
    other_max: float
    other_step: float
    max_known_misroute_pct: float


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration settings for the calibrated model candidate."""

    method: str
    cv: int


def _load_model_config() -> dict[str, object]:
    """Load the dedicated model TOML config once."""
    with MODEL_CONFIG_FILE.open("rb") as handle:
        return tomllib.load(handle)


_MODEL_CONFIG = _load_model_config()


def _load_default_thresholds() -> dict[str, float]:
    """Load model threshold defaults from the dedicated TOML config."""
    thresholds = _MODEL_CONFIG.get("thresholds", {})
    return {
        "auto_accept": float(thresholds["auto_accept"]),
        "manual_review": float(thresholds["manual_review"]),
        "other": float(thresholds["other"]),
    }


def _load_threshold_search_config() -> ThresholdSearchConfig:
    """Load threshold-tuning bounds and guardrails from model config."""
    search = _MODEL_CONFIG.get("threshold_selection", {})
    return ThresholdSearchConfig(
        other_min=float(search["other_min"]),
        other_max=float(search["other_max"]),
        other_step=float(search["other_step"]),
        max_known_misroute_pct=float(search["max_known_misroute_pct"]),
    )


def _load_calibration_config() -> CalibrationConfig:
    """Load calibration defaults for calibrated model candidates."""
    calibration = _MODEL_CONFIG.get("calibration", {})
    return CalibrationConfig(
        method=str(calibration["method"]),
        cv=int(calibration["cv"]),
    )


DEFAULT_THRESHOLDS: dict[str, float] = _load_default_thresholds()
THRESHOLD_SEARCH = _load_threshold_search_config()
CALIBRATION = _load_calibration_config()
