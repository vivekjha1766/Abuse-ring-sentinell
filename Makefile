# ============================================================
# Abuse-Ring Sentinel — developer task runner
# Every target is a thin wrapper around an explicit command, so
# you can always run the underlying command directly instead
# (useful on Windows where `make` may be unavailable).
# ============================================================

.DEFAULT_GOAL := help
.PHONY: help install install-all sync lint format typecheck test test-leakage test-fast \
        coverage clean data api frontend docker-up docker-down migrate check

## ---- Environment -------------------------------------------------

install:  ## Create venv and install core + dev dependencies
	uv sync

install-all:  ## Install every optional extra (simulator, ml, graph, api) + dev
	uv sync --extra all

sync: install-all  ## Alias for install-all

## ---- Quality gates ----------------------------------------------

lint:  ## Lint with ruff
	uv run ruff check .

format:  ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Static type check with mypy
	uv run mypy simulator ml graph backend

test:  ## Run the full pytest suite
	uv run pytest

test-fast:  ## Run everything except slow tests
	uv run pytest -m "not slow"

test-leakage:  ## Run ONLY the point-in-time / leakage guards
	uv run pytest -m leakage -v

coverage:  ## Run tests with coverage report
	uv run pytest --cov --cov-report=term-missing --cov-report=html

check: lint typecheck test  ## Full local gate: lint + types + tests

## ---- Data & models ----------------------------------------------

data:  ## Generate the synthetic dataset (Checkpoint 1+)
	uv run python scripts/generate_dataset.py

## ---- Services ----------------------------------------------------

api:  ## Run the FastAPI backend with reload (Checkpoint 4+)
	uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

migrate:  ## Apply Alembic migrations (Checkpoint 4+)
	uv run alembic -c backend/alembic.ini upgrade head

frontend:  ## Run the Vite dev server (Checkpoint 5+)
	cd frontend && npm run dev

## ---- Docker ------------------------------------------------------

docker-up:  ## Start the full stack (Checkpoint 7+)
	docker compose up --build

docker-down:  ## Stop the full stack and remove volumes
	docker compose down -v

## ---- Housekeeping ------------------------------------------------

clean:  ## Remove caches and build artifacts (keeps generated data)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
