.DEFAULT_GOAL := help
.PHONY: help install sample-model bootstrap full-model train evaluate serve test lint format typecheck docker-build docker-run

DATA_DIR ?= data/trellis_assessment_ds
MODEL ?= models/document_classifier.joblib

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (incl. dev tools)
	uv sync --dev

sample-model: ## Train a smoke model on the committed sample corpus (no download)
	uv run python -m src.train --data-dir data/sample --output $(MODEL) --report-dir reports

bootstrap: ## Get full dataset (DATA_SRC=local folder/zip, else download), train, evaluate
	./scripts/bootstrap.sh

full-model: ## Train + evaluate on a dataset already at $(DATA_DIR) (no download)
	uv run python -m src.train --data-dir $(DATA_DIR) --output $(MODEL) --report-dir reports
	uv run python -m src.evaluate --data-dir $(DATA_DIR) --model-path $(MODEL) --report-dir reports --loco

train: ## Train on the full dataset ($(DATA_DIR))
	uv run python -m src.train --data-dir $(DATA_DIR) --output $(MODEL) --report-dir reports

evaluate: ## Regenerate reports + tune thresholds (add ARGS=--loco for the OOD probe)
	uv run python -m src.evaluate --data-dir $(DATA_DIR) --model-path $(MODEL) --report-dir reports $(ARGS)

serve: ## Run the API locally
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

test: ## Run the fast unit tests
	uv run pytest -m "not integration"

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

typecheck: ## Type-check with basedpyright
	uv run basedpyright

docker-build: ## Build the container (bakes a smoke model from data/sample)
	docker build -t document-classifier .

docker-run: ## Run the container
	docker run --rm -p 8000:8000 document-classifier
