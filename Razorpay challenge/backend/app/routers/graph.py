from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.graph_service import graph_for_transaction, rings

router = APIRouter()


@router.get("/graph/{transaction_id}")
def get_graph(transaction_id: str, radius: int = Query(1, ge=1, le=3)) -> dict:
    result = graph_for_transaction(transaction_id, radius=radius)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown transaction_id: {transaction_id}")
    return result


@router.get("/rings")
def get_rings(
    min_accounts: int = Query(2, ge=2, le=50), limit: int = Query(10, ge=1, le=100)
) -> dict:
    return {"rings": rings(min_accounts=min_accounts, limit=limit)}
