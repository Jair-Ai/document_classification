"""Centralized application settings backed by Dynaconf.

Values are loaded from ``settings.toml`` at the project root and can be
overridden with environment variables, e.g. ``MODEL_PATH`` or
``API__MAX_DOCUMENT_LENGTH``.
"""

from pathlib import Path

from dynaconf import Dynaconf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

settings = Dynaconf(
    settings_files=[str(_PROJECT_ROOT / "settings.toml")],
    envvar_prefix=False,
    environments=False,
)
