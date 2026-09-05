"""GET /stats: dataset counts plus the FROZEN Checkpoint 2H heldout results.

The heldout numbers below are not computed by this service. They are the
numbers reported by scripts/evaluate_heldout.py against the sealed heldout
split, evaluated exactly once, and are served here read-only for the
dashboard -- clearly labelled as a frozen, one-time evaluation, never as a
live production metric.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.data_store import load_demo_data

router = APIRouter()

FROZEN_HELDOUT_RESULTS = {
    "label": "Frozen heldout evaluation \u2014 Checkpoint 2H (evaluated once, not a live metric)",
    "models": [
        {
            "model_type": "majority",
            "pr_auc": 0.0508,
            "roc_auc": 0.5000,
            "precision": 0.0508,
            "recall": 1.0000,
            "f1": 0.0966,
            "total_error_cost": 399660,
        },
        {
            "model_type": "logistic_regression",
            "pr_auc": 0.5864,
            "roc_auc": 0.9023,
            "precision": 0.1547,
            "recall": 0.8810,
            "f1": 0.2623,
            "total_error_cost": 179490,
        },
        {
            "model_type": "xgboost",
            "pr_auc": 0.7430,
            "roc_auc": 0.9603,
            "precision": 0.2460,
            "recall": 0.9270,
            "f1": 0.3888,
            "total_error_cost": 107535,
        },
    ],
}


@router.get("/stats")
def get_stats() -> dict:
    data = load_demo_data()
    return {
        "dataset": {
            "transactions_loaded": len(data.history),
            "accounts_loaded": len(data.accounts),
            "is_synthetic_fallback": data.is_synthetic_fallback,
            "note": (
                "Loaded from data/splits/{warmup,train,validation} only; "
                "heldout is never read by this service."
            ),
        },
        "heldout_evaluation": FROZEN_HELDOUT_RESULTS,
    }
