# Document Classifier

> WIP — assembled at integration.

A document classification service: a TF-IDF–based scikit-learn model behind a
FastAPI endpoint, with confidence-based routing to an `other` fallback for
out-of-distribution documents.

## Planned sections

- Project overview
- Setup & quickstart (uv sync, training, running the API)
- Model decision summary
- `other` fallback & threshold trade-off
- Confidence range report
- Scaling to millions of documents
- API endpoints & response statuses
- Quality gates
- Jupyter workflow
- Future improvements
