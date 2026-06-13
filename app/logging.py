"""Logging configuration controlled by Dynaconf settings."""

import json
import logging
import logging.config
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(settings: Any) -> None:
    """Configure root/application logging from Dynaconf settings.

    Production defaults to JSON logs. Development can switch to a human
    readable format with ``ENV_FOR_DYNACONF=development`` or override
    individual values with ``LOGGING__LEVEL`` and ``LOGGING__JSON``.
    """
    level = str(settings.logging.level).upper()
    json_logs = bool(settings.logging.json)
    formatter_name = "json" if json_logs else "console"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": "app.logging.JsonFormatter"},
                "console": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": formatter_name,
                },
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
            "loggers": {
                "app": {"level": level, "propagate": True},
                "uvicorn": {"level": level, "propagate": True},
                "uvicorn.error": {"level": level, "propagate": True},
                "uvicorn.access": {"level": level, "propagate": True},
            },
        }
    )
