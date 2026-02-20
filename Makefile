# ──────────────────────────────────────────────────────────────────────────
# SCRA – Self-Correcting Regulatory Agent  (Makefile)
#
# NOTE: This Makefile uses bash/Unix commands and is designed for Linux/macOS.
# On Windows, run the Python commands directly in PowerShell instead:
#   conda run -n scra --no-capture-output pip install -r requirements.txt
#   conda run -n scra --no-capture-output pytest tests/ -v --tb=short
#   conda run -n scra --no-capture-output uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000
# ──────────────────────────────────────────────────────────────────────────
SHELL   := /bin/bash
PYTHON  := conda run --no-capture-output -n scra python
PIP     := conda run --no-capture-output -n scra pip
PYTEST  := conda run --no-capture-output -n scra pytest
UVICORN := conda run --no-capture-output -n scra uvicorn

.DEFAULT_GOAL := help

# ── Environment ──────────────────────────────────────────────────────────
.PHONY: env
env:  ## Create / update conda env & install deps
	conda create -n scra python=3.11 -y || true
	$(PIP) install -r requirements.txt

# ── Linting / Formatting ────────────────────────────────────────────────
.PHONY: lint
lint:  ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/

.PHONY: format
format:  ## Run ruff formatter
	$(PYTHON) -m ruff format src/ tests/

# ── Tests ────────────────────────────────────────────────────────────────
.PHONY: test
test:  ## Run unit tests (no live API calls)
	$(PYTEST) tests/ -v --tb=short -x

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=term-missing

.PHONY: test-live
test-live:  ## Run integration tests (requires API keys)
	RUN_LIVE_TESTS=1 $(PYTEST) tests/test_integration.py -v --tb=short

# ── Evaluation ───────────────────────────────────────────────────────────
.PHONY: eval
eval:  ## Run golden dataset evaluation against running server
	$(PYTHON) -m evals.run_eval --output evals/results/latest.json

.PHONY: eval-baseline
eval-baseline:  ## Run eval and save as baseline for regression checks
	$(PYTHON) -m evals.run_eval --output evals/results/baseline.json
	@echo "Baseline saved to evals/results/baseline.json"

.PHONY: eval-compare
eval-compare:  ## Compare latest eval against baseline
	$(PYTHON) -m evals.compare_runs evals/results/baseline.json evals/results/latest.json

# ── Ingestion ────────────────────────────────────────────────────────────
.PHONY: ingest
ingest:  ## Download & ingest EU AI Act into ChromaDB
	$(PYTHON) -c "import asyncio; from src.container import Container; c=Container(); asyncio.run(c.ingestion_pipeline.run())"

# ── Server ───────────────────────────────────────────────────────────────
.PHONY: serve
serve:  ## Start FastAPI dev server on port 8000
	$(UVICORN) src.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

# ── Utilities ────────────────────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache chroma_data test_chroma_*

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
