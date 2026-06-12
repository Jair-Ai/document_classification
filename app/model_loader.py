"""Model bundle loading for the classification API.

The joblib bundle is loaded once at application startup (see the
lifespan handler in ``app.main``) and cached on the application state,
so request handlers never touch the filesystem. A missing or corrupt
artifact must not crash the service: ``load_bundle`` returns ``None``
and the API runs in degraded mode (503 on classification, health check
reporting ``model_loaded: false``) until a valid model is deployed.
"""

import logging
import os
from pathlib import Path
from typing import Any

import joblib

from app.config import settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Keys a bundle must provide for the API to serve predictions.
REQUIRED_BUNDLE_KEYS = ("model", "target_names", "trained_labels")


def resolve_model_path() -> Path:
    """Return the model artifact path, resolved at call time.

    The ``MODEL_PATH`` environment variable takes precedence over the
    ``model_path`` entry in ``settings.toml``, so deployments and tests
    can point at a different artifact without editing config files.
    Relative paths are resolved against the project root so the app
    behaves the same regardless of the working directory.
    """
    raw = os.environ.get("MODEL_PATH") or str(settings.model_path)
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def load_bundle() -> dict[str, Any] | None:
    """Load and validate the model bundle, or return ``None``.

    Any failure (missing file, unreadable artifact, wrong structure) is
    logged and swallowed so the application starts in degraded mode
    instead of crash-looping at boot.
    """
    path = resolve_model_path()
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        logger.error("Model artifact not found at %s; starting without a model", path)
        return None
    except Exception:
        logger.exception("Failed to load model artifact at %s", path)
        return None

    if not isinstance(bundle, dict):
        logger.error("Model artifact at %s is not a bundle dict; ignoring it", path)
        return None

    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
    if missing:
        logger.error("Model bundle at %s is missing required keys: %s", path, missing)
        return None

    logger.info(
        "Loaded model bundle from %s (model_type=%s, labels=%d)",
        path,
        bundle.get("model_type", "unknown"),
        len(bundle["trained_labels"]),
    )
    return bundle
