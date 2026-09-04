"""Point-in-time temporal burst features.

For transaction ``T``, the current rolling count and volume are measured over
``[T - window, T)``.  Their baseline is the distribution of the same rolling
measure at earlier account transaction timestamps.  Each historical anchor is
queried through ``TransactionHistory.as_of(anchor)``, so neither the anchored row,
equal-timestamp peers, nor future rows can affect a score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ml.feature_registry import TEMPORAL_WINDOWS
from ml.point_in_time import MIN_HISTORY_FOR_DEVIATION, MIN_STD, TransactionHistory


def _value(row: Any, field: str) -> Any:
    return row[field] if isinstance(row, Mapping) else getattr(row, field)


def _zscore(value: float, baseline: list[float]) -> float:
    """Return a finite z-score or the documented cold-start default of zero."""
    if len(baseline) < MIN_HISTORY_FOR_DEVIATION:
        return 0.0
    mean = math.fsum(baseline) / len(baseline)
    variance = math.fsum((item - mean) ** 2 for item in baseline) / (len(baseline) - 1)
    std = math.sqrt(max(0.0, variance))
    if std <= MIN_STD:
        return 0.0
    result = (value - mean) / std
    return float(result) if math.isfinite(result) else 0.0


def temporal_features(history: TransactionHistory, transaction: Any) -> dict[str, float]:
    """Compute the six specified account temporal z-scores for one transaction.

    The result contains only derived numeric feature values.  Transaction labels,
    attack IDs, and identifiers are never returned or used as model inputs; the
    account ID is used solely to scope historical rows.
    """
    timestamp = _value(transaction, "timestamp")
    account_id = _value(transaction, "account_id")
    view = history.as_of(timestamp)
    prior_rows = view.rows(key="account_id", value=account_id)
    anchors = tuple(sorted({row["timestamp"] for row in prior_rows}))
    result: dict[str, float] = {}

    for label, window in TEMPORAL_WINDOWS.items():
        current_count = float(view.count(window, key="account_id", value=account_id))
        current_volume = float(view.amount_sum(window, key="account_id", value=account_id))
        count_baseline: list[float] = []
        volume_baseline: list[float] = []
        for anchor in anchors:
            anchored_view = history.as_of(anchor)
            count_baseline.append(
                float(anchored_view.count(window, key="account_id", value=account_id))
            )
            volume_baseline.append(
                float(anchored_view.amount_sum(window, key="account_id", value=account_id))
            )
        result[f"temporal_zscore_count_{label}"] = _zscore(current_count, count_baseline)
        result[f"temporal_zscore_volume_{label}"] = _zscore(current_volume, volume_baseline)
    return result
