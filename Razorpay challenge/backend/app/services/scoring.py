"""Scores one already-known (non-heldout) transaction with the frozen model.

Fits nothing. Never reads heldout. A thin inference wrapper around the
unmodified ml.features.build_feature_matrix and the frozen Checkpoint 2G
XGBoost artifact's own predict_proba, applying that run's already-selected
validation threshold unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.features import build_feature_matrix
from ml.preprocessing import assert_model_matrix_safe

from backend.app.config import risk_band
from backend.app.services.artifacts import load_artifacts
from backend.app.services.data_store import load_demo_data


@dataclass(frozen=True)
class ScoreResult:
    transaction_id: str
    fraud_probability: float
    risk_score: int
    risk_band: str
    decision: str
    threshold: float
    feature_vector: dict


def score_transaction(transaction_id: str) -> ScoreResult:
    data = load_demo_data()
    transaction = data.get(transaction_id)
    if transaction is None:
        raise KeyError(transaction_id)

    # Same call pattern as scripts/train_baseline_models.py /
    # scripts/evaluate_heldout.py: the target row is folded into its own
    # history, and PIT safety (ts < as_of) is enforced unchanged inside
    # ml/features.py and ml/point_in_time.py.
    matrix = build_feature_matrix(
        [transaction], history_transactions=data.history, accounts=data.accounts
    )
    assert_model_matrix_safe(matrix, context="api feature matrix")

    artifacts = load_artifacts()
    probability = float(artifacts.xgboost_model.predict_proba(matrix)[0])
    score = round(probability * 100)
    decision = "flag" if probability >= artifacts.threshold else "allow"

    return ScoreResult(
        transaction_id=transaction_id,
        fraud_probability=probability,
        risk_score=score,
        risk_band=risk_band(score),
        decision=decision,
        threshold=artifacts.threshold,
        feature_vector=matrix.iloc[0].to_dict(),
    )
