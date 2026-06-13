# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: train a small "smoke" model on the committed sample corpus
# (data/sample) so the runtime image is functional out of the box — a clean
# clone can `docker build && docker run` and immediately classify, with no
# external dataset download. Production swaps in the full-accuracy model via
# the MODEL_PATH env var / a mounted volume (see README "Models: two tiers").
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY config ./config
COPY data/sample ./data/sample

# Train the baked-in smoke model from the sample corpus. Deterministic
# (fixed seed in src/train.py), so the image build is reproducible.
RUN uv run --no-sync python -m src.train \
    --data-dir data/sample \
    --output models/document_classifier.joblib \
    --report-dir /tmp/build-reports

# ---------------------------------------------------------------------------
# Runtime stage: API service only. Slim — no training deps, no sample data.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV ENV_FOR_DYNACONF=production \
    MODEL_PATH=/app/models/document_classifier.joblib \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY config ./config
COPY src ./src
COPY README.md ./

# Bake the smoke model trained in the build stage so the container is
# immediately functional. Override at deploy time with a real model by
# mounting it and setting MODEL_PATH, e.g.:
#   docker run -v "$PWD/models:/app/models:ro" document-classifier
COPY --from=builder /app/models/document_classifier.joblib ./models/document_classifier.joblib

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
