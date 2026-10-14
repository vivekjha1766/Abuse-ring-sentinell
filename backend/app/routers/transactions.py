from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.data_store import load_demo_data

router = APIRouter()


@router.get("/transaction/{transaction_id}")
def get_transaction(transaction_id: str) -> dict:
    data = load_demo_data()
    transaction = data.get(transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail=f"unknown transaction_id: {transaction_id}")
    return {
        "transaction_id": transaction.transaction_id,
        "timestamp": transaction.timestamp.isoformat(),
        "account_id": transaction.account_id,
        "customer_id": transaction.customer_id,
        "merchant_id": transaction.merchant_id,
        "device_id": transaction.device_id,
        "ip_id": transaction.ip_id,
        "amount": transaction.amount,
        "currency": transaction.currency,
        "channel": transaction.channel,
        "transaction_type": transaction.transaction_type,
        "status": transaction.status,
    }


@router.get("/transactions")
def list_transactions(limit: int = Query(10, ge=1, le=200)) -> dict:
    """A small, deterministic list of valid transaction ids for demo/sample use.

    Returns the first `limit` transaction ids from the already-loaded
    non-heldout dataset (or the synthetic fallback fixture), in the same
    order every time -- used by the frontend to populate the sample
    transaction dropdown. Performs no scoring or filtering and never reads
    heldout data.
    """
    data = load_demo_data()
    return {"transaction_ids": data.sample_ids(limit)}
