"""Runtime configuration for the fraud detection demo API.

Loads only the FROZEN Checkpoint 2G/2H artifacts and non-heldout data. This
module does not modify simulator/, ml/, or any frozen artifact; it only
names where to find them.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"

#: The frozen Checkpoint 2G experiment_id this service serves predictions
#: from. Must name a run whose artifacts already exist under artifacts/.
EXPERIMENT_ID = os.environ.get("SENTINEL_EXPERIMENT_ID", "")

#: Splits this service may ever read transaction data from. "heldout" is
#: deliberately never here, mirroring scripts/train_baseline_models.py and
#: scripts/evaluate_heldout.py's own PERMITTED_SPLITS guards.
PERMITTED_DATA_SPLITS = ("warmup", "train", "validation")

#: Fixed risk-band mapping over the model's own 0-100 probability score.
#: Not a new ML score -- purely a presentation banding of fraud_probability.
RISK_LOW_MAX = 30
RISK_MEDIUM_MAX = 70


def risk_band(score_0_100: float) -> str:
    if score_0_100 < RISK_LOW_MAX:
        return "Low"
    if score_0_100 < RISK_MEDIUM_MAX:
        return "Medium"
    return "High"
