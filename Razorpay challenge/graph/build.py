"""Build an investigative NetworkX graph from non-heldout transactions.

For investigation/visualization only. Never used to alter the frozen
XGBoost score, threshold, or any evaluation number. Built from whatever
transactions the caller supplies -- callers in this codebase only ever
pass TRAIN + VALIDATION (+ warmup) data; heldout is never read here.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def build_investigation_graph(transactions: list[Any]) -> nx.Graph:
    """A simple heterogeneous graph: customer/account/device/ip/merchant nodes.

    Node ids are ``(type, value)`` tuples so distinct entity types never
    collide even if their raw ids happen to share a string.
    """
    graph = nx.Graph()
    for txn in transactions:
        account_node = ("account", txn.account_id)
        device_node = ("device", txn.device_id)
        ip_node = ("ip", txn.ip_id)
        customer_node = ("customer", txn.customer_id)

        graph.add_node(account_node, type="account", id=txn.account_id)
        graph.add_node(device_node, type="device", id=txn.device_id)
        graph.add_node(ip_node, type="ip", id=txn.ip_id)
        graph.add_node(customer_node, type="customer", id=txn.customer_id)

        graph.add_edge(customer_node, account_node, relation="OWNS")
        graph.add_edge(account_node, device_node, relation="USES")
        graph.add_edge(account_node, ip_node, relation="LOGGED_IN_FROM")

        if txn.merchant_id:
            merchant_node = ("merchant", txn.merchant_id)
            graph.add_node(merchant_node, type="merchant", id=txn.merchant_id)
            graph.add_edge(account_node, merchant_node, relation="TRANSACTS_WITH")
    return graph
