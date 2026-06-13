"""Optional API-key enforcement for classification endpoints."""

import logging
import secrets
from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

AUTH_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def enforce_api_key(request: Request, settings: Any) -> JSONResponse | None:
    """Return an auth error response when API-key protection rejects a request.

    API-key auth is intentionally optional. It can be enabled with
    ``SECURITY__API_KEY_ENABLED=true`` and configured with
    ``SECURITY__API_KEY`` / ``SECURITY__API_KEY_HEADER``. Health and
    documentation endpoints remain public so orchestrators and local
    developers can inspect the service without a credential.
    """
    if request.url.path in AUTH_EXEMPT_PATHS:
        return None

    if not bool(settings.security.api_key_enabled):
        return None

    expected_key = str(settings.security.api_key)
    header_name = str(settings.security.api_key_header)
    if not expected_key:
        logger.error("API-key authentication is enabled but SECURITY__API_KEY is empty")
        return JSONResponse(
            status_code=500,
            content={"detail": "API key authentication is misconfigured"},
        )

    supplied_key = request.headers.get(header_name)
    if supplied_key is None or not secrets.compare_digest(supplied_key, expected_key):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return None
