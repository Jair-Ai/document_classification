"""FastAPI application exposing the document classification service.

The model bundle is loaded once at startup via the lifespan handler and
cached on ``app.state``; request handlers are read-only against it, so
the process is stateless and can be scaled horizontally with multiple
workers (e.g. ``uvicorn app.main:app --workers 4``).
"""

import codecs
import hashlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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

DEFAULT_CORS_METHODS = "GET, POST, OPTIONS"


class RequestBodyTooLargeError(Exception):
    """Raised when a request body exceeds the configured byte budget."""


def _configured_int(name: str, default: int) -> int:
    """Read an integer API setting with a fallback for older configs."""
    return int(getattr(settings.api, name, default))


def _upload_chunk_size_bytes() -> int:
    """Chunk size used when incrementally reading uploaded files."""
    return _configured_int("upload_chunk_size_bytes", 64 * 1024)


def _multipart_overhead_bytes() -> int:
    """Multipart framing allowance added to request byte budgets."""
    return _configured_int("multipart_overhead_bytes", 64 * 1024)


def _max_file_upload_bytes() -> int:
    """Maximum decoded endpoint file payload in raw bytes."""
    return _configured_int(
        "max_file_upload_bytes",
        int(settings.api.max_document_length) * 4,
    )


def _max_request_bytes(scope: Scope) -> int:
    """Return the byte limit for the incoming request body."""
    if scope.get("path") == "/classify-file":
        return _max_file_upload_bytes() + _multipart_overhead_bytes()
    return _configured_int(
        "max_request_bytes",
        int(settings.api.max_batch_size) * int(settings.api.max_document_length) * 4
        + _multipart_overhead_bytes(),
    )


def _security_headers() -> dict[str, str]:
    """Static response headers for the JSON API surface."""
    return {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "X-Frame-Options": "DENY",
    }


def _cors_allowed_origins() -> set[str]:
    """Return the configured browser origins allowed to call the API."""
    origins = getattr(settings.security, "cors_allowed_origins", [])
    if isinstance(origins, str):
        if not origins.strip():
            return set()
        return {origin.strip() for origin in origins.split(",") if origin.strip()}
    return {str(origin).strip() for origin in origins if str(origin).strip()}


def _cors_headers(origin: str | None) -> dict[str, str]:
    """Return simple-response CORS headers for an allowed origin."""
    if origin is None or origin not in _cors_allowed_origins():
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Vary": "Origin",
    }


def _preflight_headers(origin: str) -> dict[str, str]:
    """Return preflight headers for an allowed browser origin."""
    allowed_headers = {"Authorization", "Content-Type"}
    api_key_header = str(getattr(settings.security, "api_key_header", "X-API-Key")).strip()
    if api_key_header:
        allowed_headers.add(api_key_header)

    return {
        **_security_headers(),
        **_cors_headers(origin),
        "Access-Control-Allow-Methods": DEFAULT_CORS_METHODS,
        "Access-Control-Allow-Headers": ", ".join(sorted(allowed_headers)),
    }


class RequestSizeLimitMiddleware:
    """Reject oversized HTTP request bodies by Content-Length or streamed bytes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = _max_request_bytes(scope)
        content_length = _content_length(scope)
        if content_length is not None and content_length > max_bytes:
            await self._send_413(scope, receive, send, max_bytes)
            return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > max_bytes:
                    raise RequestBodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLargeError:
            if response_started:
                raise
            await self._send_413(scope, receive, send, max_bytes)

    @staticmethod
    async def _send_413(scope: Scope, receive: Receive, send: Send, max_bytes: int) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds {max_bytes} bytes"},
        )
        response.headers.update(_security_headers())
        await response(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name == b"content-length":
            try:
                return int(raw_value)
            except ValueError:
                return None
    return None


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
app.add_middleware(RequestSizeLimitMiddleware)


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

    origin = request.headers.get("origin")
    is_preflight = (
        request.method == "OPTIONS"
        and origin is not None
        and request.headers.get("access-control-request-method") is not None
    )

    if is_preflight:
        if origin in _cors_allowed_origins():
            response = Response(status_code=204)
            response.headers.update(_preflight_headers(origin))
        else:
            response = JSONResponse(status_code=400, content={"detail": "CORS origin not allowed"})
            response.headers.update(_security_headers())
    else:
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

    response.headers.update(_security_headers())
    response.headers.update(_cors_headers(origin))
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


async def _read_upload_text(file: UploadFile) -> str:
    """Decode an upload incrementally while enforcing byte and character caps."""
    max_bytes = _max_file_upload_bytes()
    max_chars = int(settings.api.max_document_length)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    parts: list[str] = []
    total_bytes = 0
    total_chars = 0

    while chunk := await file.read(_upload_chunk_size_bytes()):
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds {max_bytes} bytes",
            )

        text_chunk = decoder.decode(chunk)
        total_chars += len(text_chunk)
        if total_chars > max_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Decoded file exceeds {max_chars} characters",
            )
        parts.append(text_chunk)

    tail = decoder.decode(b"", final=True)
    total_chars += len(tail)
    if total_chars > max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Decoded file exceeds {max_chars} characters",
        )
    parts.append(tail)

    return "".join(parts)


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

    text = await _read_upload_text(file)

    return _classify(text, top_k, request)
