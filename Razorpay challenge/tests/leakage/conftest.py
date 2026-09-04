"""Fixtures for the point-in-time and split-policy guards.

Deliberately no generated dataset. These tests assert properties of the safety
mechanism, and a hand-built list of a dozen rows shows a boundary-exclusion bug far
more clearly than 156,352 real ones — while running in milliseconds instead of the
13-40 seconds a generation costs. The at-scale confirmation on the real dataset is
``scripts/audit_ml_readiness.py``, not this suite.

The Checkpoint 1 regression suite in ``tests/simulator/`` owns dataset properties
and is not duplicated here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from simulator.transactions import Transaction

#: All timestamps are offsets from this instant, so every assertion reads as
#: "minute 30" rather than as a wall-clock date.
ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _txn(
    index: int,
    *,
    minutes: float = 0.0,
    amount: float = 100.0,
    account: str = "acc_1",
    customer: str = "cus_1",
    device: str = "dev_1",
    ip: str = "ip_1",
    merchant: str | None = "mer_1",
    lat: float = 10.0,
    lon: float = 20.0,
    status: str = "completed",
    is_fraud: bool = False,
) -> Transaction:
    return Transaction(
        transaction_id=f"txn_{index:05d}",
        timestamp=ORIGIN + timedelta(minutes=minutes),
        account_id=account,
        customer_id=customer,
        device_id=device,
        ip_id=ip,
        amount=amount,
        currency="USD",
        channel="web",
        transaction_type="purchase",
        lat=lat,
        lon=lon,
        status=status,
        merchant_id=merchant,
        beneficiary_id=None,
        is_fraud=is_fraud,
        attack_id="atk_train_fake_merchant_ring_0000" if is_fraud else None,
    )


@pytest.fixture
def make_txn():
    """Factory for a single transaction; every field has a usable default."""
    return _txn


@pytest.fixture
def origin() -> datetime:
    return ORIGIN
