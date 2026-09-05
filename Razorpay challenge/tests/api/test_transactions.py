"""Tests for GET /transactions -- sample transaction ids for the demo UI.

Bug-fix coverage only: the frontend's sample dropdown was empty because it
read from GET /alerts (only ever non-empty when the frozen model actually
flags something in the sampled window). This checks the new, dedicated
endpoint returns a small, deterministic, valid set of ids instead.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.mark.api
def test_list_transactions_returns_valid_ids() -> None:
    response = client.get("/transactions?limit=5")
    assert response.status_code == 200

    body = response.json()
    ids = body["transaction_ids"]
    assert 1 <= len(ids) <= 5

    # Every returned id must resolve on the existing single-transaction endpoint.
    for transaction_id in ids:
        detail = client.get(f"/transaction/{transaction_id}")
        assert detail.status_code == 200
        assert detail.json()["transaction_id"] == transaction_id


@pytest.mark.api
def test_list_transactions_is_deterministic() -> None:
    first = client.get("/transactions?limit=10").json()
    second = client.get("/transactions?limit=10").json()
    assert first == second


@pytest.mark.api
def test_list_transactions_default_limit() -> None:
    response = client.get("/transactions")
    assert response.status_code == 200
    assert len(response.json()["transaction_ids"]) <= 10


@pytest.mark.api
def test_list_transactions_rejects_out_of_range_limit() -> None:
    response = client.get("/transactions?limit=0")
    assert response.status_code == 422
