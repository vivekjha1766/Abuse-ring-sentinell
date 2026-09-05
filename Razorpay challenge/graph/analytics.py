"""Interpretable, investigation-only analytics over the demo graph.

Offline/demo analytics only. Where a component's label concentration is
reported, non-heldout labels are used solely to report a countable,
human-readable signal for investigators -- never to fit, tune, or select a
threshold for the frozen XGBoost model, which this module never touches.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def shared_device_account_count(graph: nx.Graph, device_id: str) -> int:
    node = ("device", device_id)
    if node not in graph:
        return 0
    return sum(1 for neighbor in graph.neighbors(node) if neighbor[0] == "account")


def shared_ip_account_count(graph: nx.Graph, ip_id: str) -> int:
    node = ("ip", ip_id)
    if node not in graph:
        return 0
    return sum(1 for neighbor in graph.neighbors(node) if neighbor[0] == "account")


def connected_components_summary(
    graph: nx.Graph,
    transactions_by_account: dict[str, list[Any]],
    *,
    min_accounts: int = 2,
) -> list[dict]:
    """Ring/community candidates: components spanning >= min_accounts accounts.

    Every field is a plain count over the component's own members -- number
    of accounts, devices, ips, merchants, transactions -- plus, where
    non-heldout labels are available, the fraction of those transactions
    marked fraudulent. This is a demo/investigation signal, not a model
    output.
    """
    summaries: list[dict] = []
    for component in nx.connected_components(graph):
        accounts = {n[1] for n in component if n[0] == "account"}
        if len(accounts) < min_accounts:
            continue
        devices = {n[1] for n in component if n[0] == "device"}
        ips = {n[1] for n in component if n[0] == "ip"}
        merchants = {n[1] for n in component if n[0] == "merchant"}

        component_transactions = [
            t for a in accounts for t in transactions_by_account.get(a, [])
        ]
        transaction_count = len(component_transactions)
        fraud_count = sum(1 for t in component_transactions if getattr(t, "is_fraud", False))

        summaries.append(
            {
                "component_id": "-".join(sorted(accounts)[:3]),
                "account_count": len(accounts),
                "device_count": len(devices),
                "ip_count": len(ips),
                "merchant_count": len(merchants),
                "transaction_count": transaction_count,
                "fraud_concentration": (
                    fraud_count / transaction_count if transaction_count else 0.0
                ),
            }
        )
    summaries.sort(key=lambda s: s["account_count"], reverse=True)
    return summaries
