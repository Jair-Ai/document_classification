"""Tests for Dynaconf environment settings and logging setup."""

import json
import logging

from app.config import create_settings
from app.logging import JsonFormatter, configure_logging


def test_production_is_default_environment(monkeypatch) -> None:
    """The service defaults to production settings when no env is selected."""
    monkeypatch.delenv("ENV_FOR_DYNACONF", raising=False)

    config = create_settings()

    assert config.current_env == "production"
    assert config.logging.level == "INFO"
    assert config.logging.json is True


def test_development_environment_switches_logging_defaults(
    monkeypatch,
) -> None:
    """ENV_FOR_DYNACONF selects environment-specific settings."""
    monkeypatch.setenv("ENV_FOR_DYNACONF", "development")

    config = create_settings()

    assert config.logging.level == "DEBUG"
    assert config.logging.json is False


def test_nested_environment_variables_override_logging(monkeypatch) -> None:
    """Nested Dynaconf env vars override file-based environment config."""
    monkeypatch.setenv("ENV_FOR_DYNACONF", "production")
    monkeypatch.setenv("LOGGING__LEVEL", "DEBUG")
    monkeypatch.setenv("LOGGING__JSON", "false")

    config = create_settings()

    assert config.logging.level == "DEBUG"
    assert config.logging.json is False


def test_json_formatter_emits_structured_log_line() -> None:
    """Structured dict messages become JSON log fields."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.requests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg={"request_id": "abc", "status_code": 200},
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.requests"
    assert payload["request_id"] == "abc"
    assert payload["status_code"] == 200


def test_configure_logging_uses_configured_level(monkeypatch) -> None:
    """configure_logging applies the effective Dynaconf log level."""
    monkeypatch.setenv("ENV_FOR_DYNACONF", "production")
    monkeypatch.setenv("LOGGING__LEVEL", "WARNING")

    configure_logging(create_settings())

    assert logging.getLogger().level == logging.WARNING
