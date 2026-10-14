"""Strict point-in-time safety and label isolation tests for the graph layer (Checkpoint 3A).

Verifies that:
1. Future transactions cannot alter an earlier graph snapshot or query.
2. An event occurring exactly at T is excluded (strict < T).
3. Future edges cannot change earlier shared-device or shared-IP counts.
4. Future-created nodes cannot appear in an earlier graph snapshot.
5. Online graph construction and analytics do not read or depend on is_fraud or attack_id.
6. Offline label-based evaluation is strictly decoupled from online serving paths.
7. Existing graph behavior on legitimate historical-only inputs remains structurally correct.
8. Caching cannot reuse future-inclusive graph state for an earlier timestamp query.
9. Query performance meets the <50 ms per-transaction SLA.

All fixtures are lightweight and synthetic; heldout data is never loaded.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from graph.analytics import (
    connected_components_summary,
    evaluate_offline_ring_fraud_concentration,
    shared_device_account_count,
    shared_ip_account_count,
)
from graph.build import PointInTimeGraphIndex, graph_as_of
from simulator.transactions import Transaction

pytestmark = pytest.mark.graph

ORIGIN = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _make_txn(
    txn_id: str,
    *,
    minutes: float = 0.0,
    account: str = "acc_1",
    customer: str = "cus_1",
    device: str = "dev_1",
    ip: str = "ip_1",
    merchant: str | None = "mer_1",
    amount: float = 100.0,
    is_fraud: bool = False,
    attack_id: str | None = None,
) -> Transaction:
    return Transaction(
        transaction_id=txn_id,
        timestamp=ORIGIN + timedelta(minutes=minutes),
        account_id=account,
        customer_id=customer,
        device_id=device,
        ip_id=ip,
        amount=amount,
        currency="USD",
        channel="web",
        transaction_type="purchase",
        lat=10.0,
        lon=20.0,
        status="completed",
        merchant_id=merchant,
        beneficiary_id=None,
        is_fraud=is_fraud,
        attack_id=attack_id,
    )


# -----------------------------------------------------------------------------
# Case 1: Future transaction cannot change an earlier graph query
# -----------------------------------------------------------------------------
def test_future_transaction_cannot_change_earlier_graph_query() -> None:
    t1 = _make_txn("txn_1", minutes=10, account="acc_1", device="dev_1")
    t2_future = _make_txn("txn_2", minutes=20, account="acc_2", device="dev_1")

    as_of = ORIGIN + timedelta(minutes=15)

    graph_past_only = graph_as_of([t1], as_of=as_of)
    graph_with_future = graph_as_of([t1, t2_future], as_of=as_of)

    assert set(graph_past_only.nodes) == set(graph_with_future.nodes)
    assert set(graph_past_only.edges) == set(graph_with_future.edges)

    index = PointInTimeGraphIndex([t1, t2_future])
    index_graph = index.graph_as_of(as_of)
    assert set(index_graph.nodes) == set(graph_past_only.nodes)
    assert set(index_graph.edges) == set(graph_past_only.edges)


# -----------------------------------------------------------------------------
# Case 2: An event exactly at T is excluded
# -----------------------------------------------------------------------------
def test_event_exactly_at_t_is_excluded() -> None:
    t_exact = ORIGIN + timedelta(minutes=10)
    t1 = _make_txn("txn_1", minutes=10, account="acc_1", device="dev_1", ip="ip_1")

    # At exactly t_exact: transaction must be excluded (ts < as_of)
    g_at_t = graph_as_of([t1], as_of=t_exact)
    assert len(g_at_t.nodes) == 0
    assert len(g_at_t.edges) == 0

    index = PointInTimeGraphIndex([t1])
    assert index.shared_device_account_count("dev_1", as_of=t_exact) == 0
    assert index.shared_ip_account_count("ip_1", as_of=t_exact) == 0
    ego_at_t = index.ego_graph_as_of("acc_1", as_of=t_exact)
    assert ego_at_t.number_of_edges() == 0

    # 1 microsecond later: transaction is included
    t_after = t_exact + timedelta(microseconds=1)
    g_after = graph_as_of([t1], as_of=t_after)
    assert ("account", "acc_1") in g_after
    assert ("device", "dev_1") in g_after
    assert index.shared_device_account_count("dev_1", as_of=t_after) == 1


# -----------------------------------------------------------------------------
# Case 3: A future edge cannot change earlier shared-device or shared-IP counts
# -----------------------------------------------------------------------------
def test_future_edge_cannot_change_earlier_shared_counts() -> None:
    t1 = _make_txn("txn_1", minutes=10, account="acc_1", device="dev_shared", ip="ip_shared")
    t2_future = _make_txn(
        "txn_2", minutes=20, account="acc_2", device="dev_shared", ip="ip_shared"
    )

    index = PointInTimeGraphIndex([t1, t2_future])

    as_of_mid = ORIGIN + timedelta(minutes=15)
    assert index.shared_device_account_count("dev_shared", as_of=as_of_mid) == 1
    assert index.shared_ip_account_count("ip_shared", as_of=as_of_mid) == 1

    as_of_after = ORIGIN + timedelta(minutes=25)
    assert index.shared_device_account_count("dev_shared", as_of=as_of_after) == 2
    assert index.shared_ip_account_count("ip_shared", as_of=as_of_after) == 2


# -----------------------------------------------------------------------------
# Case 4: A future-created node cannot appear in an earlier graph snapshot
# -----------------------------------------------------------------------------
def test_future_created_node_cannot_appear_in_earlier_snapshot() -> None:
    t1 = _make_txn(
        "txn_1",
        minutes=10,
        account="acc_1",
        customer="cus_1",
        device="dev_1",
        ip="ip_1",
        merchant="mer_1",
    )
    t2_future = _make_txn(
        "txn_2",
        minutes=20,
        account="acc_new",
        customer="cus_new",
        device="dev_new",
        ip="ip_new",
        merchant="mer_new",
    )

    index = PointInTimeGraphIndex([t1, t2_future])
    as_of_mid = ORIGIN + timedelta(minutes=15)
    g = index.graph_as_of(as_of_mid)

    assert ("account", "acc_1") in g
    assert ("account", "acc_new") not in g
    assert ("customer", "cus_new") not in g
    assert ("device", "dev_new") not in g
    assert ("ip", "ip_new") not in g
    assert ("merchant", "mer_new") not in g


# -----------------------------------------------------------------------------
# Case 5: Online graph construction and analytics do not read or depend on labels
# -----------------------------------------------------------------------------
def test_online_graph_does_not_read_or_contain_labels() -> None:
    t_clean = _make_txn("txn_1", minutes=5, account="acc_1", is_fraud=False, attack_id=None)
    t_fraud = _make_txn(
        "txn_1", minutes=5, account="acc_1", is_fraud=True, attack_id="atk_simulated_01"
    )

    as_of = ORIGIN + timedelta(minutes=10)

    g_clean = graph_as_of([t_clean], as_of=as_of)
    g_fraud = graph_as_of([t_fraud], as_of=as_of)

    # Topology and attributes must be identical
    assert set(g_clean.nodes) == set(g_fraud.nodes)
    assert set(g_clean.edges) == set(g_fraud.edges)

    for _node, data in g_fraud.nodes(data=True):
        assert "is_fraud" not in data
        assert "attack_id" not in data

    for _u, _v, data in g_fraud.edges(data=True):
        assert "is_fraud" not in data
        assert "attack_id" not in data

    # Online summaries contain no label fields
    summaries = connected_components_summary(g_fraud)
    assert len(summaries) == 0  # <2 accounts

    # Multi-account component
    t2_fraud = _make_txn("txn_2", minutes=6, account="acc_2", device=t_fraud.device_id)
    g_multi = graph_as_of([t_fraud, t2_fraud], as_of=as_of)
    multi_summaries = connected_components_summary(g_multi, min_accounts=2)

    assert len(multi_summaries) == 1
    assert "fraud_concentration" not in multi_summaries[0]
    assert "is_fraud" not in multi_summaries[0]
    assert set(multi_summaries[0].keys()) == {
        "component_id",
        "account_count",
        "device_count",
        "ip_count",
        "merchant_count",
        "transaction_count",
    }


# -----------------------------------------------------------------------------
# Case 6: Offline label-based evaluation is clearly separated from serving
# -----------------------------------------------------------------------------
def test_offline_label_evaluation_separation() -> None:
    t1 = _make_txn("txn_1", minutes=5, account="acc_1", device="dev_ring")
    t2 = _make_txn("txn_2", minutes=6, account="acc_2", device="dev_ring")

    as_of = ORIGIN + timedelta(minutes=10)
    g = graph_as_of([t1, t2], as_of=as_of)

    by_account = {"acc_1": [t1], "acc_2": [t2]}
    online_summaries = connected_components_summary(g, by_account, min_accounts=2)
    assert len(online_summaries) == 1
    assert "fraud_concentration" not in online_summaries[0]

    # Offline evaluation requires explicit ground_truth_labels
    ground_truth = {"txn_1": True, "txn_2": False}
    offline_summaries = evaluate_offline_ring_fraud_concentration(
        online_summaries,
        transactions_by_account=by_account,
        ground_truth_labels=ground_truth,
    )
    assert len(offline_summaries) == 1
    assert "fraud_concentration" in offline_summaries[0]
    assert offline_summaries[0]["fraud_concentration"] == 0.5

    # Verify backend services does not import evaluate_offline_ring_fraud_concentration
    import backend.app.services.graph_service as gs

    assert not hasattr(gs, "evaluate_offline_ring_fraud_concentration")


# -----------------------------------------------------------------------------
# Case 7: Existing graph behavior on legitimate historical-only inputs remains correct
# -----------------------------------------------------------------------------
def test_existing_graph_behavior_remains_correct() -> None:
    t1 = _make_txn(
        "txn_1",
        minutes=5,
        account="acc_1",
        customer="cus_1",
        device="dev_1",
        ip="ip_1",
        merchant="mer_1",
    )
    as_of = ORIGIN + timedelta(minutes=10)

    g = graph_as_of([t1], as_of=as_of)

    assert ("account", "acc_1") in g
    assert ("customer", "cus_1") in g
    assert ("device", "dev_1") in g
    assert ("ip", "ip_1") in g
    assert ("merchant", "mer_1") in g

    assert g.has_edge(("customer", "cus_1"), ("account", "acc_1"))
    assert g[("customer", "cus_1")][("account", "acc_1")]["relation"] == "OWNS"

    assert g.has_edge(("account", "acc_1"), ("device", "dev_1"))
    assert g[("account", "acc_1")][("device", "dev_1")]["relation"] == "USES"

    assert g.has_edge(("account", "acc_1"), ("ip", "ip_1"))
    assert g[("account", "acc_1")][("ip", "ip_1")]["relation"] == "LOGGED_IN_FROM"

    assert g.has_edge(("account", "acc_1"), ("merchant", "mer_1"))
    assert g[("account", "acc_1")][("merchant", "mer_1")]["relation"] == "TRANSACTS_WITH"

    assert shared_device_account_count(g, "dev_1") == 1
    assert shared_ip_account_count(g, "ip_1") == 1


# -----------------------------------------------------------------------------
# Case 8: Caching cannot reuse future-inclusive graph state for an earlier timestamp
# -----------------------------------------------------------------------------
def test_caching_cannot_leak_across_timestamps() -> None:
    t1 = _make_txn("txn_1", minutes=10, account="acc_1", device="dev_shared")
    t2 = _make_txn("txn_2", minutes=20, account="acc_2", device="dev_shared")
    t3 = _make_txn("txn_3", minutes=30, account="acc_3", device="dev_shared")

    index = PointInTimeGraphIndex([t1, t2, t3])

    as_of_future = ORIGIN + timedelta(minutes=25)  # sees t1 and t2
    as_of_past = ORIGIN + timedelta(minutes=15)  # sees t1 only

    # 1. Query future first (populates cache)
    future_graph = index.graph_as_of(as_of_future)
    assert ("account", "acc_2") in future_graph
    assert index.shared_device_account_count("dev_shared", as_of=as_of_future) == 2

    # 2. Query past subsequently
    past_graph = index.graph_as_of(as_of_past)
    assert ("account", "acc_2") not in past_graph
    assert ("account", "acc_1") in past_graph
    assert index.shared_device_account_count("dev_shared", as_of=as_of_past) == 1

    # 3. Independent fresh graph at past instant must match exactly
    fresh_past_graph = graph_as_of([t1, t2, t3], as_of=as_of_past)
    assert set(past_graph.nodes) == set(fresh_past_graph.nodes)
    assert set(past_graph.edges) == set(fresh_past_graph.edges)


# -----------------------------------------------------------------------------
# Case 9: Ego network equivalence and performance SLA (<50 ms target)
# -----------------------------------------------------------------------------
def test_ego_network_and_performance_sla() -> None:
    # Generate 1,000 synthetic chronological transactions spanning 50 accounts and 10 shared devices
    txns = []
    for i in range(1000):
        txns.append(
            _make_txn(
                f"txn_{i:04d}",
                minutes=float(i),
                account=f"acc_{i % 50}",
                device=f"dev_{i % 10}",
                ip=f"ip_{i % 20}",
            )
        )

    index = PointInTimeGraphIndex(txns)
    query_ts = ORIGIN + timedelta(minutes=500.5)

    # Benchmark entity degree lookups (100 iterations)
    t0 = time.perf_counter()
    for _i in range(100):
        dev_cnt = index.shared_device_account_count("dev_3", as_of=query_ts)
        ip_cnt = index.shared_ip_account_count("ip_7", as_of=query_ts)
    t1 = time.perf_counter()
    avg_degree_ms = ((t1 - t0) / 100) * 1000

    assert dev_cnt > 0
    assert ip_cnt > 0
    assert avg_degree_ms < 1.0, f"Average degree lookup was {avg_degree_ms:.3f} ms"

    # Benchmark ego graph construction (50 iterations)
    t2 = time.perf_counter()
    for _ in range(50):
        ego = index.ego_graph_as_of("acc_10", as_of=query_ts, radius=1)
    t3 = time.perf_counter()
    avg_ego_ms = ((t3 - t2) / 50) * 1000

    assert len(ego.nodes) > 0
    assert avg_ego_ms < 10.0, f"Average ego graph construction was {avg_ego_ms:.3f} ms"
