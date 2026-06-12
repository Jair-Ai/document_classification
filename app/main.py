"""FastAPI application exposing the document classification service.

The model bundle is loaded once at startup via the lifespan handler and
cached on ``app.state``; request handlers are read-only against it, so
the process is stateless and can be scaled horizontally with multiple
workers (e.g. ``uvicorn app.main:app --workers 4``).
"""

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response

from app.model_loader import load_bundle
from app.schemas import ClassificationRequest, ClassificationResponse
from src.predict import predict_text

logger = logging.getLogger(__name__)
request_logger = logging.getLogger("app.requests")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the model bundle once at startup and release it on shutdown."""
    application.state.model_bundle = load_bundle()
    yield
    application.state.model_bundle = None


app = FastAPI(
    title="Document Classification API",
    description="Classifies news documents into topic categories.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Emit one structured (JSON) log line per request.

    The line carries a request ID, latency, and any classification
    metadata the handler attached to ``request.state`` — never the
    document text itself, both for privacy and to keep log volume sane
    at high throughput.
    """
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    record: dict[str, Any] = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    record.update(getattr(request.state, "log_fields", {}))
    request_logger.info(json.dumps(record))

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness probe; also reports whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": request.app.state.model_bundle is not None,
    }


@app.post("/classify_document", response_model=ClassificationResponse)
def classify_document(payload: ClassificationRequest, request: Request) -> ClassificationResponse:
    """Classify a single document and route it by confidence.

    Returns 503 while the model is unavailable, 400 for whitespace-only
    text, and 500 (with the real error logged server-side only) if
    inference fails unexpectedly.
    """
    bundle: dict[str, Any] | None = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    text = payload.document_text
    if not text.strip():
        raise HTTPException(status_code=400, detail="document_text must not be empty")

    # The schema caps top_k at the configured maximum; additionally clamp
    # it to the number of labels the model was actually trained on.
    top_k = min(payload.top_k, len(bundle["trained_labels"]))

    try:
        result = predict_text(text, bundle, top_k=top_k)
    except Exception:
        logger.exception(
            "Unexpected error during classification (request_id=%s)",
            getattr(request.state, "request_id", "-"),
        )
        raise HTTPException(
            status_code=500, detail="Unexpected classification error"
        ) from None

    # Classification metadata for the per-request log line. The text is
    # summarized as length + hash; raw content is never logged.
    request.state.log_fields = {
        "label": result["label"],
        "confidence": round(result["confidence"], 4),
        "decision": result["decision"],
        "text_length": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }

    return ClassificationResponse(message="Classification successful", **result)
