# Document Classification API

FastAPI service that classifies news documents into topic categories
using a trained scikit-learn pipeline. The model bundle is loaded once
at startup; every prediction is routed through a confidence policy that
auto-accepts high-confidence labels, flags mid-confidence ones for
review, and falls back to `"other"` when the model is unsure.

## Running the service

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive docs (Swagger UI) are served at `http://localhost:8000/docs`.

## Endpoints

### `GET /health`

Liveness probe. Always returns 200 — including when the model failed to
load — so orchestrators can distinguish "process down" from "model
unavailable".

```json
{"status": "ok", "model_loaded": true}
```

`model_loaded` is `false` when the service is running in degraded mode
(model artifact missing or unreadable); classification requests then
return 503 until a valid artifact is deployed and the service restarts.

### `POST /classify_document`

Classifies a single document passed as JSON.

**Request body**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `document_text` | string | required, 1–100,000 chars | Raw document text to classify |
| `top_k` | int | optional, 1–10, default 3 | Number of (label, confidence) pairs to return |

`top_k` is additionally clamped to the number of labels the deployed
model was trained on.

**Example request**

```json
{
  "document_text": "Dollar hits new low versus euro as traders weigh rate cuts...",
  "top_k": 3
}
```

**Example response (200)**

```json
{
  "message": "Classification successful",
  "label": "business",
  "raw_label": "business",
  "confidence": 0.93,
  "decision": "auto_accept",
  "top_k": [
    {"label": "business", "confidence": 0.93},
    {"label": "politics", "confidence": 0.04},
    {"label": "technology", "confidence": 0.02}
  ]
}
```

**Response fields**

| Field | Meaning |
|---|---|
| `label` | Final label after confidence routing; may be `"other"` |
| `raw_label` | The model's argmax label, before routing |
| `confidence` | Maximum predicted probability, in [0, 1] |
| `decision` | `auto_accept`, `review_recommended`, `manual_review`, or `fallback_other` |
| `top_k` | Top-k labels with confidences, sorted descending |

Decision routing uses the thresholds shipped inside the model bundle
(defaults: auto-accept at ≥ 0.90, review band down to 0.70, fallback to
`"other"` below 0.55).

## Response status codes

| Status | When | Body |
|---|---|---|
| 200 | Classification succeeded | `ClassificationResponse` (see example above) |
| 400 | Whitespace-only `document_text` | `{"detail": "document_text must not be empty"}` |
| 422 | Schema validation failure (missing field, length, top_k out of range) | FastAPI validation detail |
| 503 | Model artifact missing/unloadable | `{"detail": "Model is unavailable"}` |
| 500 | Unexpected inference failure | `{"detail": "Unexpected classification error"}` |

## Example curl

```bash
curl -X POST http://localhost:8000/classify_document \
  -H "Content-Type: application/json" \
  -d '{"document_text": "Dollar hits new low versus euro...", "top_k": 3}'
```

## Operations notes

### Configuration

Settings live in `settings.toml` at the repo root and are loaded via
Dynaconf (`app/config.py`). Any value can be overridden with an
environment variable — no code or config-file change needed per
environment:

```bash
MODEL_PATH=/opt/models/classifier_v2.joblib   # model artifact location
API__MAX_DOCUMENT_LENGTH=50000                # nested [api] keys use __
API__DEFAULT_TOP_K=5
```

`MODEL_PATH` is resolved at load time (env var first, then
`settings.toml`), so deployments can point at a new artifact without
rebuilding.

### Workers and statelessness

The application holds no per-request state: the model bundle is loaded
once in the startup lifespan and is read-only afterwards. The service
can therefore be scaled horizontally — multiple uvicorn workers
(`--workers 4`) or multiple replicas behind a load balancer — with no
coordination, sticky sessions, or shared cache required. Each worker
loads its own copy of the model; size memory accordingly.

### Degraded mode

A missing or corrupt model artifact does not crash the service. The
loader logs the error, `/health` keeps returning 200 with
`model_loaded: false`, and classification requests return 503. This
avoids crash-loops at boot and lets the platform alert on the health
payload instead.

### Logging policy

One structured (JSON) log line is emitted per request: request ID,
method, path, status code, latency in ms, and — for classifications —
the predicted label, confidence, decision, text length, and a SHA-256
hash of the text. The raw document text is **never** logged: it may
contain sensitive content, and at millions of documents per day it
would dominate log volume. The length + hash pair is enough to
correlate duplicates and debug payload issues. Every response carries
an `X-Request-ID` header for client-side correlation.

### Payload cap rationale

`document_text` is capped at 100,000 characters (configurable via
`API__MAX_DOCUMENT_LENGTH`). The cap bounds per-request memory and
vectorization CPU, keeping tail latency predictable and preventing a
single oversized payload from starving other requests. News articles
fit comfortably under it; clients with longer documents should chunk
them or raise the limit explicitly for their environment.

### Future improvements

- **Rate limiting** — per-client request quotas (e.g. token bucket at
  the gateway or via middleware) to protect the service from abusive or
  runaway clients.
- **Authentication** — API keys or OAuth2 client credentials; the
  endpoint is currently unauthenticated and must sit behind a trusted
  gateway.
- **Batch endpoint** — accept a list of documents per request to
  amortize HTTP overhead for bulk backfills.
- **Model/version metadata** — expose the loaded `model_type` and an
  artifact version in `/health` for deployment verification.
- **Metrics** — Prometheus counters/histograms (request rate, latency,
  decision mix) alongside the structured logs.
