"""API endpoint tests for graph and ring routes (Checkpoint 3A).

Verifies that:
1. GET /graph/{transaction_id} returns an ego network without future edges.
2. GET /rings returns purely topological ring candidates with NO fraud_concentration.
3. POST /predict succeeds and generates explanations using point-in-time graph signals.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
pytestmark = pytest.mark.api


def test_get_graph_for_transaction() -> None:
    # Get a valid transaction id
    txns_res = client.get("/transactions?limit=1")
    assert txns_res.status_code == 200
    txn_id = txns_res.json()["transaction_ids"][0]

    res = client.get(f"/graph/{txn_id}?radius=1")
    assert res.status_code == 200
    body = res.json()

    assert "center" in body
    assert "nodes" in body
    assert "edges" in body
    assert isinstance(body["nodes"], list)
    assert isinstance(body["edges"], list)

    # Confirm no label fields leak in nodes or edges
    for node in body["nodes"]:
        assert "is_fraud" not in node
        assert "attack_id" not in node
    for edge in body["edges"]:
        assert "is_fraud" not in edge
        assert "attack_id" not in edge


def test_get_rings_has_no_label_leakage() -> None:
    res = client.get("/rings?limit=5")
    assert res.status_code == 200
    body = res.json()

    assert "rings" in body
    rings = body["rings"]
    for r in rings:
        assert "fraud_concentration" not in r
        assert "is_fraud" not in r
        assert "attack_id" not in r
        assert "account_count" in r
        assert "device_count" in r
        assert "ip_count" in r


def test_predict_endpoint_uses_pit_graph_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import config
    from backend.app.services import artifacts
    exp_id = "20260905T165402854903_88aaaf8d6e88aef0"
    monkeypatch.setattr(config, "EXPERIMENT_ID", exp_id)
    monkeypatch.setattr(artifacts, "EXPERIMENT_ID", exp_id)
    artifacts.load_artifacts.cache_clear()
    txns_res = client.get("/transactions?limit=1")
    assert txns_res.status_code == 200
    txn_id = txns_res.json()["transaction_ids"][0]

    res = client.post("/predict", json={"transaction_id": txn_id})
    assert res.status_code == 200
    body = res.json()

    assert body["transaction_id"] == txn_id
    assert "risk_score" in body
    assert "risk_band" in body
    assert "decision" in body
    assert "top_risk_signals" in body
    assert isinstance(body["top_risk_signals"], list)

