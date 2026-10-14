"""Loads the frozen Checkpoint 2G XGBoost artifact and its own threshold.

Read-only. Never fits, never retrains, never touches heldout. The
threshold applied here is read verbatim from that run's own experiment
report (selected on VALIDATION when Checkpoint 2G ran) -- it is not
recomputed or reselected anywhere in this service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib

from backend.app.config import ARTIFACTS_ROOT, EXPERIMENT_ID


class ArtifactsNotConfiguredError(RuntimeError):
    """SENTINEL_EXPERIMENT_ID is unset, or its artifacts are missing."""


@dataclass(frozen=True)
class FrozenArtifacts:
    experiment_id: str
    xgboost_model: Any
    threshold: float


@lru_cache(maxsize=1)
def load_artifacts() -> FrozenArtifacts:
    if not EXPERIMENT_ID:
        raise ArtifactsNotConfiguredError(
            "SENTINEL_EXPERIMENT_ID is not set. Point it at the experiment_id "
            "of a completed scripts/train_baseline_models.py run before "
            "starting the API, e.g.: export SENTINEL_EXPERIMENT_ID=<id>"
        )
    model_path = ARTIFACTS_ROOT / "models" / f"xgboost_{EXPERIMENT_ID}.joblib"
    report_path = ARTIFACTS_ROOT / "reports" / f"{EXPERIMENT_ID}.json"
    for path in (model_path, report_path):
        if not path.exists():
            raise ArtifactsNotConfiguredError(f"missing frozen artifact: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    xgboost_entry = next(m for m in report["models"] if m["model_type"] == "xgboost")
    threshold = float(xgboost_entry["threshold"]["threshold"])

    return FrozenArtifacts(
        experiment_id=EXPERIMENT_ID,
        xgboost_model=joblib.load(model_path),
        threshold=threshold,
    )
