"""Request/response models for the document classification API."""

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    """Request body for ``POST /classify_document``."""

    document_text: str = Field(..., min_length=1, max_length=100_000)
    top_k: int = Field(default=3, ge=1, le=10)


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
