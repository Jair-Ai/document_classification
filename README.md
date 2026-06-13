# Document Classifier

A production-style document classification service for topical text
documents. It trains a sparse TF-IDF + LogisticRegression model, wraps
the trained pipeline in a small joblib bundle, and serves predictions
through FastAPI with confidence-based routing to an `other` fallback.

The dataset contains ten trained labels:

`business`, `entertainment`, `food`, `graphics`, `historical`,
`medical`, `politics`, `space`, `sport`, `technology`.

The source folder typo `technologie` is normalized to `technology`.
The `other` folder is not trained as a class; it is used only as an
out-of-distribution holdout to test fallback behavior.

## Setup

```bash
uv sync
```

Train the model bundle:

```bash
uv run python -m src.train \
  --data-dir ../trellis_assessment_ds \
  --output models/document_classifier.joblib \
  --report-dir reports
```

Generate evaluation reports and tune thresholds:

```bash
uv run python -m src.evaluate \
  --data-dir ../trellis_assessment_ds \
  --model-path models/document_classifier.joblib \
  --report-dir reports
```

Run the API:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive API docs are available at `http://localhost:8000/docs`.

By default the API is configured for server-to-server use behind a load
balancer. Browser origins are denied unless
`SECURITY__CORS_ALLOWED_ORIGINS` is set to an explicit allowlist.

For bulk synchronous inference, use `POST /classify_documents` instead
of sending one HTTP request per document. The request is capped at 100
documents by default (`API__MAX_BATCH_SIZE`) so memory and latency stay
bounded.

Request-body guardrails are configurable via Dynaconf environment
variables such as `API__MAX_FILE_UPLOAD_BYTES`,
`API__UPLOAD_CHUNK_SIZE_BYTES`, `API__MULTIPART_OVERHEAD_BYTES`, and
`API__MAX_REQUEST_BYTES`.

Run with Docker:

```bash
docker build -t document-classifier .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  document-classifier
```

An AWS deployment plan is documented in `infra/README.md`. It outlines
ECS Fargate, ECR, ALB, Secrets Manager, CloudWatch Logs, S3 model
artifacts, autoscaling, and a queue-backed worker path for
million-document backfills.

## Model Decision Summary

Model selection in `notebooks/02_model_experiments.ipynb` uses 5-fold
stratified cross-validation for a stable comparison on a small corpus.
The reproducible training path then uses a fixed stratified 70/15/15
split: train fits the selected model, validation tunes confidence
thresholds, and test reports final held-out metrics.

| Model | CV accuracy | CV macro F1 | Median ms/doc | Batch docs/sec | Size MB |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogisticRegression | 0.9788 | 0.9789 | 0.26 | 6792.06 | 3.28 |
| TF-IDF + MultinomialNB | 0.9778 | 0.9776 | 0.43 | 6721.01 | 5.80 |
| TF-IDF + Calibrated LinearSVC | 0.9819 | 0.9818 | 1.92 | 6187.01 | 13.36 |

The calibrated LinearSVC is slightly ahead on CV macro F1, but the
difference is small and it is slower and larger. MultinomialNB is fast
but less useful for confidence routing. LogisticRegression is the best
production baseline here: simple, compact, fast, and accurate.

Final held-out LogisticRegression test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.9933 |
| Macro F1 | 0.9933 |
| Weighted F1 | 0.9933 |

Detailed artifacts:

- `reports/model_comparison_cv.csv`
- `reports/classification_report.txt`
- `reports/confusion_matrix.csv`
- `notebooks/03_error_analysis.ipynb`

## Fallback Thresholds

The model predicts one of the ten known labels. `src.predict.predict_text`
then applies confidence thresholds from the bundle:

| Threshold | Value |
|---|---:|
| `auto_accept` | 0.90 |
| `manual_review` | 0.70 |
| `other` | 0.15 |

The low `other` threshold is intentional. Many correct predictions have
diffuse probability mass, so a high fallback threshold would route known
documents to `other` too aggressively.

Validation threshold trade-off:

| `other` threshold | Known docs routed to `other` | Known misroute % | Other holdout caught |
|---:|---:|---:|---:|
| 0.10 | 0/149 | 0.00% | 0/6 |
| 0.15 | 3/149 | 2.01% | 6/6 |
| 0.20 | 14/149 | 9.40% | 6/6 |
| 0.35 | 71/149 | 47.65% | 6/6 |
| 0.55 | 129/149 | 86.58% | 6/6 |

The chosen threshold, 0.15, catches all six `other` holdout files while
keeping known-class misroutes under 5% on validation. The OOD evidence
is useful but anecdotal because the holdout has only six files.

## Confidence Ranges

Validation confidence buckets:

| Confidence range | Total | Correct | Accuracy |
|---|---:|---:|---:|
| [0.0, 0.5) | 118 | 114 | 0.9661 |
| [0.5, 0.6) | 18 | 18 | 1.0000 |
| [0.6, 0.7) | 11 | 11 | 1.0000 |
| [0.7, 0.8) | 1 | 1 | 1.0000 |
| [0.8, 0.9) | 1 | 1 | 1.0000 |
| [0.9, 1.0] | 0 | 0 | n/a |

Low confidence does not mean the model is usually wrong: most
validation examples are below 0.5 confidence and still reach 96.61%
accuracy. The confidence policy is a routing policy, not a claim of
perfect calibration.

## Scaling To Millions Of Documents

Measured local inference benchmark:

| Benchmark | Value |
|---|---:|
| Median single-document latency | 0.3282 ms |
| p95 single-document latency | 0.5489 ms |
| Batch throughput | 4876.28 docs/sec |
| Batch size used | 149 docs |
| Serialized bundle size | 2.07 MB |

Sparse linear inference is CPU-only, compact, and easy to scale
horizontally. The API is stateless: each worker loads its own read-only
model bundle, so additional workers or replicas can be added behind a
load balancer without coordination or sticky sessions.

For bulk backfills, a batch endpoint would improve throughput by
amortizing HTTP, JSON, vectorization, and model-call overhead. For
millions of documents, the next step would be asynchronous job
processing: queue batches through workers, write results to storage, and
use the API for job submission/status rather than one huge synchronous
request. Transformer or LLM-based approaches could be useful for harder
semantic or OOD cases, but they would be materially more expensive and
slower for this topical news-style task.

## API

### `GET /health`

Returns service liveness and model availability:

```json
{"status": "ok", "model_loaded": true}
```

The endpoint still returns 200 when the model is unavailable so
operators can distinguish process health from artifact loading.

### `POST /classify_document`

Request:

```json
{
  "document_text": "The company reported stronger quarterly earnings...",
  "top_k": 3
}
```

Response:

```json
{
  "message": "Classification successful",
  "label": "business",
  "raw_label": "business",
  "confidence": 0.1985,
  "decision": "manual_review",
  "top_k": [
    {"label": "business", "confidence": 0.1985},
    {"label": "space", "confidence": 0.1022},
    {"label": "graphics", "confidence": 0.1015}
  ]
}
```

Example curl:

```bash
curl -X POST http://localhost:8000/classify_document \
  -H "Content-Type: application/json" \
  -d '{"document_text": "The company reported stronger quarterly earnings as investors reacted to higher revenue and profit forecasts.", "top_k": 3}'
```

### `POST /classify_documents`

Batch endpoint for synchronous bulk classification. This endpoint makes
one vectorized model call for the submitted texts, reducing HTTP and
model overhead compared with many single-document calls.

Request:

```json
{
  "documents": [
    {"id": "doc-1", "document_text": "The company reported stronger quarterly earnings..."},
    {"id": "doc-2", "document_text": "The striker scored twice in the final..."}
  ],
  "top_k": 3
}
```

Response:

```json
{
  "message": "Batch classification successful",
  "total": 2,
  "results": [
    {
      "document_id": "doc-1",
      "label": "business",
      "raw_label": "business",
      "confidence": 0.1985,
      "decision": "manual_review",
      "top_k": [
        {"label": "business", "confidence": 0.1985}
      ]
    }
  ]
}
```

### `POST /classify-file`

Multipart `.txt` upload endpoint. The file is decoded as UTF-8 with
invalid bytes ignored, then classified through the same path as the JSON
endpoint.

```bash
curl -X POST "http://localhost:8000/classify-file?top_k=3" \
  -F "file=@article.txt"
```

### Response Statuses

| Status | When | Body |
|---|---|---|
| 200 | Classification succeeded | `ClassificationResponse` |
| 400 | Whitespace-only text or non-`.txt` upload | `{"detail": "document_text must not be empty"}` |
| 401 | API key missing or invalid when enabled | `{"detail": "Invalid or missing API key"}` |
| 413 | Raw request/upload exceeds configured byte limits | `{"detail": "Request body exceeds ... bytes"}` |
| 422 | Schema validation failure | FastAPI validation detail |
| 503 | Model artifact missing or unloadable | `{"detail": "Model is unavailable"}` |
| 500 | Unexpected inference failure | `{"detail": "Unexpected classification error"}` |

## Configuration

Settings live in `config/settings.toml` and are loaded with Dynaconf.
Select an environment with `ENV_FOR_DYNACONF` and override nested values
with double-underscore environment variables:

```bash
ENV_FOR_DYNACONF=production
MODEL_PATH=/opt/models/classifier_v2.joblib
API__MAX_DOCUMENT_LENGTH=50000
API__MAX_FILE_UPLOAD_BYTES=200000
API__MAX_REQUEST_BYTES=20000000
API__MAX_BATCH_SIZE=100
API__DEFAULT_TOP_K=5
LOGGING__LEVEL=DEBUG
LOGGING__JSON=false
SECURITY__API_KEY_ENABLED=true
SECURITY__API_KEY=change-me
SECURITY__CORS_ALLOWED_ORIGINS='["https://app.example.com"]'
```

`MODEL_PATH` is resolved at load time, with the environment variable
taking precedence over `config/settings.toml`.

## Logging And Operations

The API configures logging in `app/logging.py` from Dynaconf
`settings.logging.*`. Production defaults to one JSON log line per
request; development can use console logs by setting
`ENV_FOR_DYNACONF=development` or `LOGGING__JSON=false`.

Request logs include request ID, method, path, status code, latency,
predicted label, confidence, decision, text length, and SHA-256 text
hash. Raw document text is never logged.

API-key protection is optional and disabled by default for local
evaluation. Enable it in deployed environments with
`SECURITY__API_KEY_ENABLED=true` and send the key in `X-API-Key`.
`/health` remains unauthenticated for platform health checks.
`SECURITY__CORS_ALLOWED_ORIGINS` can be set to an explicit browser
origin allowlist when the API is called directly from front-end clients.
Responses also include `X-Content-Type-Options`, `Referrer-Policy`, and
`X-Frame-Options`; HSTS remains an edge concern owned by the public TLS
terminator.

A missing or corrupt model artifact does not crash the service. The app
stays up, `/health` reports `model_loaded: false`, and classification
requests return 503.

## Quality Gates

```bash
uv run ruff format . --check
uv run ruff check .
uv run basedpyright
uv run pip-audit
uv run pytest -q
```

Current gate results:

- Ruff format/check: pass
- basedpyright: pass
- pip-audit: no known vulnerabilities found
- pytest: 45 passed, 1 third-party deprecation warning from Starlette

## Jupyter Workflow

The notebooks are reviewer-facing analysis artifacts:

- `notebooks/01_dataset_exploration.ipynb`
- `notebooks/02_model_experiments.ipynb`
- `notebooks/03_error_analysis.ipynb`

Run Jupyter locally with:

```bash
uv run jupyter lab
```

Notebook outputs are committed where they support the model decision.
Source, app, and tests are covered by Ruff and basedpyright; notebooks
are excluded from Ruff because they include path setup cells and
narrative analysis rather than importable modules.

## Future Improvements

- Add rate limiting at the service or gateway layer.
- Add async job processing for million-document backfills.
- Add Terraform modules once AWS account, VPC, domain, and deployment
  constraints are known.
- Collect a larger, representative OOD set and retune fallback routing.
- Add model/version metadata to `/health`.
- Export Prometheus metrics for latency, throughput, and decision mix.
- Add Docker Compose for local API smoke testing with mounted artifacts.
