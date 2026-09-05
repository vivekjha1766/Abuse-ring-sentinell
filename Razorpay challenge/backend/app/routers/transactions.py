from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
