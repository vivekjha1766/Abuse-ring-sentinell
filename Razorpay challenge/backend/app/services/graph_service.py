"""Point-in-time graph service adapter for the API layer.

Ensures all graph queries, ego networks, and entity signals are strictly
point-in-time safe (as-of the query transaction's timestamp).
Offline label concentrations are excluded from online responses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from backend.app.services.data_store import load_demo_data
from graph.build import PointInTimeGraphIndex


@lru_cache(maxsize=1)
def _get_pit_index() -> PointInTimeGraphIndex:
    data = load_demo_data()
    return PointInTimeGraphIndex(data.history)


def entity_signals_for_transaction(transaction_id: str) -> dict | None:
    """Return point-in-time entity degree signals as of the transaction timestamp."""
    data = load_demo_data()
    transaction = data.get(transaction_id)
    if transaction is None:
        return None

    index = _get_pit_index()
    return {
        "shared_device_account_count": index.shared_device_account_count(
            transaction.device_id, as_of=transaction.timestamp
        ),
        "shared_ip_account_count": index.shared_ip_account_count(
            transaction.ip_id, as_of=transaction.timestamp
        ),
    }


def graph_for_transaction(transaction_id: str, *, radius: int = 1) -> dict | None:
    """Return the ego network around the transaction's account as of its timestamp."""
    data = load_demo_data()
    transaction = data.get(transaction_id)
    if transaction is None:
        return None

    index = _get_pit_index()
    ego = index.ego_graph_as_of(
        transaction.account_id,
        as_of=transaction.timestamp,
        radius=radius,
    )
    center = ("account", transaction.account_id)
    if center not in ego:
        return None

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


def rings(
    *,
    as_of: datetime | None = None,
    min_accounts: int = 2,
    limit: int = 10,
) -> list[dict]:
    """Return ring candidate components as of the given timestamp without fraud labels."""
    index = _get_pit_index()
    if as_of is None:
        data = load_demo_data()
        as_of = max(t.timestamp for t in data.history) if data.history else datetime.now(UTC)

    return index.rings_as_of(as_of, min_accounts=min_accounts, limit=limit)
