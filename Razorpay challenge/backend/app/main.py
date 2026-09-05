"""Fraud detection demo API.

Serves inference from the FROZEN Checkpoint 2G XGBoost artifact and
reports the FROZEN Checkpoint 2H heldout results. Never fits, tunes, or
reads heldout for anything at runtime. See README.md and
docs/SUBMISSION_RESULTS.md.

Start with (from the `Razorpay challenge/` directory):
    export SENTINEL_EXPERIMENT_ID=<your Checkpoint 2G experiment_id>
    uv run uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routers import alerts, graph, health, predict, stats, transactions

app = FastAPI(
    title="Abuse-Ring Sentinel \u2014 Fraud Detection Demo API",
    description=(
        "Inference-only demo service over the frozen Checkpoint 2G XGBoost "
        "model. Synthetic data. Heldout is never read at runtime."
    ),
    version="2I",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(transactions.router)
app.include_router(alerts.router)
app.include_router(stats.router)
app.include_router(graph.router)

_DEMO_DIR = Path(__file__).resolve().parents[2] / "frontend" / "demo"
if _DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(_DEMO_DIR), html=True), name="demo")
