from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.artifacts import ArtifactsNotConfiguredError
from backend.app.services.explain import explain
from backend.app.services.scoring import score_transaction

router = APIRouter()


class PredictRequest(BaseModel):
    transaction_id: str


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        result = score_transaction(request.transaction_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown transaction_id: {request.transaction_id}"
        ) from exc
    except ArtifactsNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    reasons = explain(result.feature_vector)

    return {
        "transaction_id": result.transaction_id,
        "fraud_probability": result.fraud_probability,
        "risk_score": result.risk_score,
        "risk_band": result.risk_band,
        "decision": result.decision,
        "threshold": result.threshold,
        "top_risk_signals": reasons,
    }
