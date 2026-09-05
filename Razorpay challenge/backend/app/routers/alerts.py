"""GET /alerts: scores a bounded, recent sample of non-heldout transactions.

Demo-appropriate, not production-scale: rescoring is done on demand over a
bounded sample rather than a precomputed store, to avoid adding a database
dependency for a submission demo. heldout is never in the sample -- it
comes from the same DemoData.history built in data_store.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services.data_store import load_demo_data
from backend.app.services.scoring import score_transaction

router = APIRouter()


@router.get("/alerts")
def list_alerts(
    limit: int = Query(20, ge=1, le=200),
    sample_size: int = Query(50, ge=1, le=2000),
) -> dict:
    data = load_demo_data()
    candidates = data.history[-sample_size:]

    alerts = []
    for transaction in candidates:
        try:
            result = score_transaction(transaction.transaction_id)
        except Exception:
            continue
        if result.decision == "flag":
            alerts.append(
                {
                    "transaction_id": result.transaction_id,
                    "fraud_probability": result.fraud_probability,
                    "risk_score": result.risk_score,
                    "risk_band": result.risk_band,
                    "decision": result.decision,
                }
            )

    alerts.sort(key=lambda a: a["fraud_probability"], reverse=True)
    return {"count": len(alerts), "sampled": len(candidates), "alerts": alerts[:limit]}
