"""Centralized application settings backed by Dynaconf.

Values are loaded from ``config/settings.toml`` and can be selected per
environment with ``ENV_FOR_DYNACONF``. Any value can be overridden with
environment variables, e.g. ``MODEL_PATH``, ``API__MAX_DOCUMENT_LENGTH``,
or ``LOGGING__LEVEL``.
"""

from pathlib import Path

from dynaconf import Dynaconf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = _PROJECT_ROOT / "config" / "settings.toml"


def create_settings() -> Dynaconf:
    """Create a Dynaconf settings object for the current environment."""
    return Dynaconf(
        settings_files=[str(SETTINGS_FILE)],
        envvar_prefix=False,
        environments=True,
        env_switcher="ENV_FOR_DYNACONF",
        env="production",
    )


settings = create_settings()
