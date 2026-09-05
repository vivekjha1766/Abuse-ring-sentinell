"""Caching adapter between graph/ and the API layer.

The investigation graph is built once from DemoData.history (never
heldout) and reused across requests, rather than rebuilt per call.
"""

from __future__ import annotations

from functools import lru_cache

import networkx as nx

from graph.analytics import (
    connected_components_summary,
    shared_device_account_count,
    shared_ip_account_count,
)
from graph.build import build_investigation_graph

from backend.app.services.data_store import load_demo_data


@lru_cache(maxsize=1)
def _graph_and_index() -> tuple[nx.Graph, dict]:
    data = load_demo_data()
    graph = build_investigation_graph(data.history)
    by_account: dict[str, list] = {}
    for txn in data.history:
        by_account.setdefault(txn.account_id, []).append(txn)
    return graph, by_account


def entity_signals_for_transaction(transaction_id: str) -> dict | None:
    data = load_demo_data()
    transaction = data.get(transaction_id)
    if transaction is None:
        return None
    graph, _ = _graph_and_index()
    return {
        "shared_device_account_count": shared_device_account_count(graph, transaction.device_id),
        "shared_ip_account_count": shared_ip_account_count(graph, transaction.ip_id),
    }


def graph_for_transaction(transaction_id: str, *, radius: int = 1) -> dict | None:
    data = load_demo_data()
    transaction = data.get(transaction_id)
    if transaction is None:
        return None
    graph, _ = _graph_and_index()
    center = ("account", transaction.account_id)
    if center not in graph:
        return None

    ego = nx.ego_graph(graph, center, radius=radius)
    return {
        "center": f"{center[0]}:{center[1]}",
        "nodes": [
            {"id": f"{n[0]}:{n[1]}", "type": n[0], "label": n[1]} for n in ego.nodes
        ],
        "edges": [
            {
                "source": f"{u[0]}:{u[1]}",
                "target": f"{v[0]}:{v[1]}",
                "relation": edge_data.get("relation", ""),
            }
            for u, v, edge_data in ego.edges(data=True)
        ],
    }


def rings(*, min_accounts: int = 2, limit: int = 10) -> list[dict]:
    graph, by_account = _graph_and_index()
    summaries = connected_components_summary(graph, by_account, min_accounts=min_accounts)
    return summaries[:limit]
