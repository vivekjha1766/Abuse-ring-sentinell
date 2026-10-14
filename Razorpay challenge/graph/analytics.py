"""Interpretable graph analytics and ring candidate extraction.

Online analytics compute purely topological graph evidence from point-in-time
graph structures without inspecting ground-truth fraud labels.
Retrospective label evaluation is isolated to explicit offline research helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx


def shared_device_account_count(graph: nx.Graph, device_id: str) -> int:
    """Count accounts directly connected to device_id in the given graph."""
    node = ("device", device_id)
    if node not in graph:
        return 0
    return sum(1 for neighbor in graph.neighbors(node) if neighbor[0] == "account")


def shared_ip_account_count(graph: nx.Graph, ip_id: str) -> int:
    """Count accounts directly connected to ip_id in the given graph."""
    node = ("ip", ip_id)
    if node not in graph:
        return 0
    return sum(1 for neighbor in graph.neighbors(node) if neighbor[0] == "account")


def connected_components_summary(
    graph: nx.Graph,
    transactions_by_account: Mapping[str, list[Any]] | None = None,
    *,
    min_accounts: int = 2,
) -> list[dict]:
    """Ring/community candidates: components spanning >= min_accounts accounts.

    Every field is a plain topological count over the component's own members:
    accounts, devices, ips, merchants, transactions.
    Online detection and investigation evidence does NOT read or report fraud labels.
    """
    summaries: list[dict] = []
    for component in nx.connected_components(graph):
        accounts = {n[1] for n in component if n[0] == "account"}
        if len(accounts) < min_accounts:
            continue
        devices = {n[1] for n in component if n[0] == "device"}
        ips = {n[1] for n in component if n[0] == "ip"}
        merchants = {n[1] for n in component if n[0] == "merchant"}

        if transactions_by_account is not None:
            component_transactions = [
                t for a in accounts for t in transactions_by_account.get(a, [])
            ]
            transaction_count = len(component_transactions)
        else:
            transaction_count = 0

        summaries.append(
            {
                "component_id": "-".join(sorted(accounts)[:3]),
                "account_count": len(accounts),
                "device_count": len(devices),
                "ip_count": len(ips),
                "merchant_count": len(merchants),
                "transaction_count": transaction_count,
            }
        )
    summaries.sort(key=lambda s: s["account_count"], reverse=True)
    return summaries


def evaluate_offline_ring_fraud_concentration(
    summaries: list[dict],
    *,
    transactions_by_account: Mapping[str, list[Any]],
    ground_truth_labels: Mapping[str, bool],
) -> list[dict]:
    """Offline research evaluation: compute retrospective fraud concentration for rings.

    This function requires explicit offline ground_truth_labels (mapping transaction_id -> bool).
    It is strictly for offline evaluation and ablation research, and must NEVER be
    imported or called by online serving paths (backend/app/services).
    """
    evaluated: list[dict] = []
    for summary in summaries:
        item = dict(summary)
        account_keys = [
            acc
            for acc, txns in transactions_by_account.items()
            if any(t.account_id == acc for t in txns)
        ]
        component_txns = []
        for acc in account_keys:
            txns = transactions_by_account.get(acc, [])
            component_txns.extend(txns)

        fraud_count = sum(
            1 for t in component_txns if ground_truth_labels.get(t.transaction_id, False)
        )
        total_txns = len(component_txns)
        item["fraud_concentration"] = (fraud_count / total_txns) if total_txns > 0 else 0.0
        evaluated.append(item)
    return evaluated
