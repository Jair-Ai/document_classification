"""Request/response models for the document classification API.

Validation limits come from ``config/settings.toml`` (see
``app.config``) so payload caps can be tuned per environment without
code changes.
"""

from pydantic import BaseModel, Field

from app.config import settings


class ClassificationRequest(BaseModel):
    """Request body for ``POST /classify_document``."""

    document_text: str = Field(
        ...,
        min_length=settings.api.min_document_length,
        max_length=settings.api.max_document_length,
    )
    top_k: int = Field(
        default=settings.api.default_top_k,
        ge=settings.api.min_top_k,
        le=settings.api.max_top_k,
    )


class BatchDocument(BaseModel):
    """One document in a batch classification request."""

    id: str | None = Field(default=None, max_length=128)
    document_text: str = Field(
        ...,
        min_length=settings.api.min_document_length,
        max_length=settings.api.max_document_length,
    )


class BatchClassificationRequest(BaseModel):
    """Request body for ``POST /classify_documents``."""

    documents: list[BatchDocument] = Field(
        ...,
        min_length=1,
        max_length=settings.api.max_batch_size,
    )
    top_k: int = Field(
        default=settings.api.default_top_k,
        ge=settings.api.min_top_k,
        le=settings.api.max_top_k,
    )


class ClassConfidence(BaseModel):
    """One (label, confidence) entry of the top-k ranking."""

    label: str
    confidence: float


class ClassificationResponse(BaseModel):
    """Successful classification result."""

    message: str
    label: str
    raw_label: str
    confidence: float
    decision: str
    top_k: list[ClassConfidence]


class BatchClassificationResult(BaseModel):
    """Successful classification result for one batched document."""

    document_id: str | None
    label: str
    raw_label: str
    confidence: float
    decision: str
    top_k: list[ClassConfidence]


class BatchClassificationResponse(BaseModel):
    """Successful batch classification response."""

    message: str
    total: int
    results: list[BatchClassificationResult]
