"""FastAPI application exposing the document classification service.

The model bundle is loaded once at startup via the lifespan handler and
cached on ``app.state``; request handlers are read-only against it, so
the process is stateless and can be scaled horizontally with multiple
workers (e.g. ``uvicorn app.main:app --workers 4``).
"""

import hashlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile

from app.config import settings
from app.logging import configure_logging
from app.model_loader import load_bundle
from app.schemas import (
    BatchClassificationRequest,
    BatchClassificationResponse,
    BatchClassificationResult,
    ClassificationRequest,
    ClassificationResponse,
)
from app.security import enforce_api_key
from src.predict import predict_batch, predict_text

configure_logging(settings)

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

    response = enforce_api_key(request, settings)
    if response is None:
        response = await call_next(request)

    record: dict[str, Any] = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    record.update(getattr(request.state, "log_fields", {}))
    request_logger.info(record)

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness probe; also reports whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": request.app.state.model_bundle is not None,
    }


def _classify(text: str, top_k: int, request: Request) -> ClassificationResponse:
    """Shared classification path for the JSON and file-upload endpoints.

    Raises 503 while the model is unavailable, 400 for whitespace-only
    text, and 500 (with the real error logged server-side only) if
    inference fails unexpectedly.
    """
    bundle: dict[str, Any] | None = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    if not text.strip():
        raise HTTPException(status_code=400, detail="document_text must not be empty")

    # The schema caps top_k at the configured maximum; additionally clamp
    # it to the number of labels the model was actually trained on.
    top_k = min(top_k, len(bundle["trained_labels"]))

    try:
        result = predict_text(text, bundle, top_k=top_k)
    except Exception:
        logger.exception(
            "Unexpected error during classification (request_id=%s)",
            getattr(request.state, "request_id", "-"),
        )
        raise HTTPException(status_code=500, detail="Unexpected classification error") from None

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


def _classify_batch(
    payload: BatchClassificationRequest,
    request: Request,
) -> BatchClassificationResponse:
    """Classify multiple documents with one vectorized model call."""
    bundle: dict[str, Any] | None = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    for index, document in enumerate(payload.documents):
        if not document.document_text.strip():
            detail: dict[str, Any] = {
                "message": "document_text must not be empty",
                "index": index,
            }
            if document.id is not None:
                detail["document_id"] = document.id
            raise HTTPException(status_code=400, detail=detail)

    top_k = min(payload.top_k, len(bundle["trained_labels"]))
    texts = [document.document_text for document in payload.documents]

    try:
        predictions = predict_batch(texts, bundle, top_k=top_k)
    except Exception:
        logger.exception(
            "Unexpected error during batch classification (request_id=%s)",
            getattr(request.state, "request_id", "-"),
        )
        raise HTTPException(status_code=500, detail="Unexpected classification error") from None

    results = [
        BatchClassificationResult(document_id=document.id, **prediction)
        for document, prediction in zip(payload.documents, predictions, strict=True)
    ]

    request.state.log_fields = {
        "batch_size": len(results),
        "total_text_length": sum(len(text) for text in texts),
        "labels": dict(Counter(result.label for result in results)),
        "decisions": dict(Counter(result.decision for result in results)),
        "text_sha256_sample": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts[:5]
        ],
    }

    return BatchClassificationResponse(
        message="Batch classification successful",
        total=len(results),
        results=results,
    )


@app.post("/classify_document", response_model=ClassificationResponse)
def classify_document(payload: ClassificationRequest, request: Request) -> ClassificationResponse:
    """Classify a single document submitted as JSON."""
    return _classify(payload.document_text, payload.top_k, request)


@app.post("/classify_documents", response_model=BatchClassificationResponse)
def classify_documents(
    payload: BatchClassificationRequest,
    request: Request,
) -> BatchClassificationResponse:
    """Classify multiple documents in one request."""
    return _classify_batch(payload, request)


@app.post("/classify-file", response_model=ClassificationResponse)
async def classify_file(
    request: Request,
    file: Annotated[UploadFile, File(description="Plain-text (.txt) document")],
    top_k: Annotated[
        int,
        Query(ge=settings.api.min_top_k, le=settings.api.max_top_k),
    ] = settings.api.default_top_k,
) -> ClassificationResponse:
    """Classify a document uploaded as a multipart .txt file.

    Only ``.txt`` uploads are accepted; the content is decoded as UTF-8
    with undecodable bytes dropped, then classified through the same
    path (and with the same length cap) as ``POST /classify_document``.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    max_length = int(settings.api.max_document_length)
    if len(text) > max_length:
        raise HTTPException(
            status_code=422,
            detail=f"Decoded file exceeds {max_length} characters",
        )

    return _classify(text, top_k, request)
