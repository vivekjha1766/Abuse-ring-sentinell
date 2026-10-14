"""Point-in-time heterogeneous graph construction.

Strict point-in-time guarantee:
    A graph or graph query as of instant `as_of` includes ONLY events
    with `event_time < as_of` (strict inequality). Events at or after
    `as_of` cannot create nodes, create edges, or alter any metric.

Node IDs are ``(type, value)`` tuples so distinct entity types never
collide even if their raw IDs happen to share a string.
Labels (``is_fraud``, ``attack_id``) are quarantined and never stored on
graph nodes or edges.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from graph.analytics import connected_components_summary


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def graph_as_of(
    transactions: Sequence[Any],
    as_of: datetime,
    *,
    accounts: Mapping[str, Any] | None = None,
) -> nx.Graph:
    """Build a heterogeneous NetworkX graph containing only events with ts < as_of.

    Strict half-open rule: `timestamp < as_of`.
    Transactions occurring at or after `as_of` are strictly excluded.
    No label fields (`is_fraud`, `attack_id`) are stored in nodes or edges.
    """
    as_of_utc = _ensure_utc(as_of)
    graph = nx.Graph()

    for txn in transactions:
        txn_ts = _ensure_utc(txn.timestamp)
        if txn_ts >= as_of_utc:
            continue

        if accounts is not None and txn.account_id in accounts:
            acc_info = accounts[txn.account_id]
            if (
                isinstance(acc_info, dict)
                and "created_at" in acc_info
                and _ensure_utc(acc_info["created_at"]) >= as_of_utc
            ):
                continue

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


def build_investigation_graph(
    transactions: Sequence[Any],
    *,
    as_of: datetime | None = None,
) -> nx.Graph:
    """Backward-compatible investigation graph builder.

    If `as_of` is provided, delegates to `graph_as_of(transactions, as_of)`.
    If `as_of` is None, includes all provided transactions up to the maximum
    timestamp.
    """
    if as_of is not None:
        return graph_as_of(transactions, as_of)
    return graph_as_of(transactions, datetime.max.replace(tzinfo=UTC))


class PointInTimeGraphIndex:
    """Chronologically sorted index for high-performance point-in-time graph queries.

    Avoids O(N) full-graph rebuilding per transaction while guaranteeing strict
    point-in-time safety (events with ts < as_of only).
    """

    def __init__(self, transactions: Sequence[Any]) -> None:
        self._sorted_txns = sorted(transactions, key=lambda t: _ensure_utc(t.timestamp))
        self._timestamps = [_ensure_utc(t.timestamp) for t in self._sorted_txns]

        # Entity-to-event indexes for microsecond-speed topological lookups:
        self._device_accounts: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        self._ip_accounts: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        self._account_txns: dict[str, list[tuple[datetime, Any]]] = defaultdict(list)

        for txn in self._sorted_txns:
            ts = _ensure_utc(txn.timestamp)
            self._device_accounts[txn.device_id].append((ts, txn.account_id))
            self._ip_accounts[txn.ip_id].append((ts, txn.account_id))
            self._account_txns[txn.account_id].append((ts, txn))

        # Snapshot cache keyed by exact as_of datetime to avoid cross-timestamp leaks
        self._graph_cache: dict[datetime, nx.Graph] = {}
        self._cache_max_size = 64

    def shared_device_account_count(self, device_id: str, as_of: datetime) -> int:
        """Count distinct accounts that used device_id strictly before as_of."""
        as_of_utc = _ensure_utc(as_of)
        entries = self._device_accounts.get(device_id)
        if not entries:
            return 0
        cutoff = bisect_left(entries, (as_of_utc,))
        return len({acc for _, acc in entries[:cutoff]})

    def shared_ip_account_count(self, ip_id: str, as_of: datetime) -> int:
        """Count distinct accounts that used ip_id strictly before as_of."""
        as_of_utc = _ensure_utc(as_of)
        entries = self._ip_accounts.get(ip_id)
        if not entries:
            return 0
        cutoff = bisect_left(entries, (as_of_utc,))
        return len({acc for _, acc in entries[:cutoff]})

    def ego_graph_as_of(
        self,
        center_account: str,
        as_of: datetime,
        *,
        radius: int = 1,
    ) -> nx.Graph:
        """Construct the ego-network around center_account using only edges with ts < as_of.

        Guarantees that no future edge or node is traversed or included.
        """
        as_of_utc = _ensure_utc(as_of)
        graph = nx.Graph()

        center_node = ("account", center_account)
        graph.add_node(center_node, type="account", id=center_account)

        acc_entries = self._account_txns.get(center_account)
        if not acc_entries:
            return graph

        cutoff = bisect_left(acc_entries, (as_of_utc,))
        prior_acc_txns = [txn for _, txn in acc_entries[:cutoff]]
        if not prior_acc_txns:
            return graph

        # Distance 1: add entities from the center account's prior transactions
        dev_nodes: set[tuple[str, str]] = set()
        ip_nodes: set[tuple[str, str]] = set()

        for txn in prior_acc_txns:
            dev_node = ("device", txn.device_id)
            ip_node = ("ip", txn.ip_id)
            cus_node = ("customer", txn.customer_id)

            graph.add_node(dev_node, type="device", id=txn.device_id)
            graph.add_node(ip_node, type="ip", id=txn.ip_id)
            graph.add_node(cus_node, type="customer", id=txn.customer_id)

            graph.add_edge(cus_node, center_node, relation="OWNS")
            graph.add_edge(center_node, dev_node, relation="USES")
            graph.add_edge(center_node, ip_node, relation="LOGGED_IN_FROM")

            dev_nodes.add(dev_node)
            ip_nodes.add(ip_node)

            if txn.merchant_id:
                mer_node = ("merchant", txn.merchant_id)
                graph.add_node(mer_node, type="merchant", id=txn.merchant_id)
                graph.add_edge(center_node, mer_node, relation="TRANSACTS_WITH")

        # Distance 2: add other accounts sharing those devices or IPs strictly before as_of
        if radius >= 2:
            for _, dev_id in dev_nodes:
                d_entries = self._device_accounts.get(dev_id, [])
                d_cutoff = bisect_left(d_entries, (as_of_utc,))
                for _, other_acc in d_entries[:d_cutoff]:
                    other_node = ("account", other_acc)
                    graph.add_node(other_node, type="account", id=other_acc)
                    graph.add_edge(other_node, ("device", dev_id), relation="USES")

            for _, ip_id in ip_nodes:
                ip_entries = self._ip_accounts.get(ip_id, [])
                ip_cutoff = bisect_left(ip_entries, (as_of_utc,))
                for _, other_acc in ip_entries[:ip_cutoff]:
                    other_node = ("account", other_acc)
                    graph.add_node(other_node, type="account", id=other_acc)
                    graph.add_edge(other_node, ("ip", ip_id), relation="LOGGED_IN_FROM")

        return graph

    def graph_as_of(self, as_of: datetime) -> nx.Graph:
        """Return the full graph strictly as of as_of."""
        as_of_utc = _ensure_utc(as_of)
        if as_of_utc in self._graph_cache:
            return self._graph_cache[as_of_utc]

        cutoff = bisect_left(self._timestamps, as_of_utc)
        sub_txns = self._sorted_txns[:cutoff]
        g = graph_as_of(sub_txns, as_of_utc)

        if len(self._graph_cache) >= self._cache_max_size:
            self._graph_cache.pop(next(iter(self._graph_cache)))
        self._graph_cache[as_of_utc] = g
        return g

    def rings_as_of(
        self,
        as_of: datetime,
        *,
        min_accounts: int = 2,
        limit: int = 10,
    ) -> list[dict]:
        """Return ring candidate components as of as_of without fraud labels."""
        as_of_utc = _ensure_utc(as_of)
        g = self.graph_as_of(as_of_utc)

        cutoff = bisect_left(self._timestamps, as_of_utc)
        sub_txns = self._sorted_txns[:cutoff]
        by_account: dict[str, list[Any]] = defaultdict(list)
        for txn in sub_txns:
            by_account[txn.account_id].append(txn)

        summaries = connected_components_summary(g, by_account, min_accounts=min_accounts)
        return summaries[:limit]
