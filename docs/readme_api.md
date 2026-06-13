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

### `POST /classify-file`

Classifies a document uploaded as a multipart `.txt` file. The response
shape is identical to `POST /classify_document`.

| Parameter | In | Constraints | Description |
|---|---|---|---|
| `file` | form-data | required, filename must end in `.txt` | Plain-text document |
| `top_k` | query | optional, 1–10, default 3 | Number of (label, confidence) pairs |

The file content is decoded as UTF-8 with undecodable bytes dropped,
then validated and classified through the same path as the JSON
endpoint: whitespace-only content returns 400, content longer than the
configured document length cap returns 422, and a non-`.txt` filename
returns 400 with `{"detail": "Only .txt files are supported"}`.

```bash
curl -X POST "http://localhost:8000/classify-file?top_k=3" \
  -F "file=@article.txt"
```

### `POST /classify_documents`

Classifies multiple documents in one synchronous request. This endpoint
uses one vectorized model call for the submitted texts, which reduces
HTTP, JSON parsing, vectorization, and model-call overhead compared with
many `POST /classify_document` requests.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `documents` | array | required, 1–100 items by default | Documents to classify |
| `documents[].id` | string/null | optional, max 128 chars | Client-provided document identifier returned in the result |
| `documents[].document_text` | string | required, 1–100,000 chars | Raw document text |
| `top_k` | int | optional, 1–10, default 3 | Number of labels per document |

```json
{
  "documents": [
    {"id": "doc-1", "document_text": "Dollar hits new low versus euro..."},
    {"id": "doc-2", "document_text": "The striker scored twice in the final..."}
  ],
  "top_k": 3
}
```

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
        {"label": "business", "confidence": 0.1985},
        {"label": "space", "confidence": 0.1022},
        {"label": "graphics", "confidence": 0.1015}
      ]
    }
  ]
}
```

The batch size cap is configured with `API__MAX_BATCH_SIZE`. For
millions of documents, use queue-backed workers and persistent result
storage; this endpoint is the bounded synchronous path.

## Response status codes

| Status | When | Body |
|---|---|---|
| 200 | Classification succeeded | `ClassificationResponse` (see example above) |
| 400 | Whitespace-only `document_text`, or non-`.txt` upload | `{"detail": "document_text must not be empty"}` / `{"detail": "Only .txt files are supported"}` |
| 401 | API key missing or invalid when enabled | `{"detail": "Invalid or missing API key"}` |
| 413 | Raw request/upload exceeds the configured byte limit | `{"detail": "Request body exceeds ... bytes"}` |
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

Settings live in `config/settings.toml` and are loaded via Dynaconf
(`app/config.py`). Select an environment with `ENV_FOR_DYNACONF`
(`development` or `production`) and override any value with an
environment variable — no code or config-file change needed per
deployment:

```bash
ENV_FOR_DYNACONF=production
MODEL_PATH=/opt/models/classifier_v2.joblib   # model artifact location
API__MAX_DOCUMENT_LENGTH=50000                # nested [api] keys use __
API__MAX_FILE_UPLOAD_BYTES=200000
API__MAX_REQUEST_BYTES=20000000
API__MAX_BATCH_SIZE=100
API__DEFAULT_TOP_K=5
LOGGING__LEVEL=DEBUG                          # nested [logging] keys
LOGGING__JSON=false
SECURITY__API_KEY_ENABLED=true                # optional API key auth
SECURITY__API_KEY=change-me
```

`MODEL_PATH` is resolved at load time (env var first, then
`config/settings.toml`), so deployments can point at a new artifact
without rebuilding.

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

Logging is configured in `app/logging.py` from Dynaconf
`settings.logging.*`. Production defaults to JSON logs; development can
switch to a console-friendly format with `ENV_FOR_DYNACONF=development`
or explicit `LOGGING__JSON=false`.

One structured log line is emitted per request: request ID, method,
path, status code, latency in ms, and — for classifications — the
predicted label, confidence, decision, text length, and a SHA-256 hash
of the text. The raw document text is **never** logged: it may contain
sensitive content, and at millions of documents per day it would
dominate log volume. The length + hash pair is enough to correlate
duplicates and debug payload issues. Every response carries an
`X-Request-ID` header for client-side correlation.

### Optional API key

API-key authentication is disabled by default for local evaluation. In a
deployed environment, enable it with:

```bash
SECURITY__API_KEY_ENABLED=true
SECURITY__API_KEY=change-me
SECURITY__API_KEY_HEADER=X-API-Key
```

Clients then include:

```bash
curl -X POST http://localhost:8000/classify_document \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"document_text": "Dollar hits new low versus euro...", "top_k": 3}'
```

`/health` remains public for platform health checks. Classification
endpoints return 401 when the API key is missing or invalid.

### Payload cap rationale

`document_text` is capped at 100,000 characters (configurable via
`API__MAX_DOCUMENT_LENGTH`). Raw request bodies are separately capped in
bytes (`API__MAX_REQUEST_BYTES`), and `.txt` uploads have a tighter file
part cap (`API__MAX_FILE_UPLOAD_BYTES`). The service rejects oversized
`Content-Length` values before parsing and also counts streamed bytes as
they arrive, so missing or chunked lengths still cannot grow without
bound. News articles fit comfortably under the defaults; clients with
longer documents should chunk them or raise the limits explicitly for
their environment.

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
