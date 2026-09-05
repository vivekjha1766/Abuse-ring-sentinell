"""Deterministic, feature-value-based explanations for a scored transaction.

Every explanation is derived directly from actual values already present
in the transaction's own computed feature vector (or, where noted, the
investigation-only graph layer) -- nothing here invents evidence that does
not trace to a real, already-computed signal.
"""

from __future__ import annotations


def explain(feature_vector: dict, *, graph_signals: dict | None = None) -> list[str]:
    reasons: list[str] = []

    if feature_vector.get("tx_velocity_1h", 0.0) >= 5.0 or abs(
        feature_vector.get("temporal_zscore_count_1h", 0.0)
    ) >= 2.0:
        reasons.append("Unusually high transaction velocity in the past hour")

    if abs(feature_vector.get("amount_deviation", 0.0)) >= 2.0:
        reasons.append("Transaction amount deviates strongly from this account's history")

    if feature_vector.get("impossible_travel", 0.0) >= 1.0:
        reasons.append(
            "Unusual geographic movement (implausible travel speed since the last transaction)"
        )

    if abs(feature_vector.get("temporal_zscore_volume_1h", 0.0)) >= 2.0:
        reasons.append("Suspicious transaction volume concentration in a short window")

    if feature_vector.get("cold_start", 0.0) >= 1.0:
        reasons.append("Limited transaction history for this account (cold start)")

    if graph_signals:
        shared_device = graph_signals.get("shared_device_account_count", 0)
        if shared_device > 1:
            reasons.append(
                f"Device shared with {shared_device - 1} other account(s)"
            )
        shared_ip = graph_signals.get("shared_ip_account_count", 0)
        if shared_ip > 1:
            reasons.append(f"IP address shared with {shared_ip - 1} other account(s)")

    if not reasons:
        reasons.append(
            "No individually unusual signal crossed its threshold; the risk score "
            "reflects the model's overall assessment across all 25 features"
        )
    return reasons
