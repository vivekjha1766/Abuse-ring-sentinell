"""Provenance for the implemented transaction feature set.

The registry is deliberately declarative: feature computation happens in
``ml.features`` and every feature has one auditable statement of its source,
historical window, aggregation, and cold-start value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping, Optional


@dataclass(frozen=True)
class FeatureSpec:
    """The provenance required for a model-eligible derived feature."""

    name: str
    source: str
    window: Optional[timedelta]
    aggregation: str
    cold_start_default: float
    point_in_time_safe: bool = True


WINDOWS: Mapping[str, timedelta] = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}

TEMPORAL_WINDOWS: Mapping[str, timedelta] = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
}


def _history_specs() -> dict[str, FeatureSpec]:
    specs: dict[str, FeatureSpec] = {}
    for label, window in WINDOWS.items():
        specs[f"tx_velocity_{label}"] = FeatureSpec(
            f"tx_velocity_{label}", "account_transactions", window, "count", 0.0
        )
        specs[f"tx_volume_{label}"] = FeatureSpec(
            f"tx_volume_{label}", "account_transactions", window, "sum", 0.0
        )
    return specs


def _temporal_specs() -> dict[str, FeatureSpec]:
    specs: dict[str, FeatureSpec] = {}
    for label, window in TEMPORAL_WINDOWS.items():
        specs[f"temporal_zscore_count_{label}"] = FeatureSpec(
            f"temporal_zscore_count_{label}",
            "account_transactions",
            window,
            "rolling_count_zscore",
            0.0,
        )
        specs[f"temporal_zscore_volume_{label}"] = FeatureSpec(
            f"temporal_zscore_volume_{label}",
            "account_transactions",
            window,
            "rolling_volume_zscore",
            0.0,
        )
    return specs


FEATURE_REGISTRY: Mapping[str, FeatureSpec] = {
    "transaction_amount": FeatureSpec("transaction_amount", "transaction", None, "none", 0.0),
    "log_transaction_amount": FeatureSpec(
        "log_transaction_amount", "transaction", None, "log1p", 0.0
    ),
    **_history_specs(),
    **_temporal_specs(),
    "amount_deviation": FeatureSpec(
        "amount_deviation", "account_transactions", None, "zscore", 0.0
    ),
    "tx_frequency_deviation": FeatureSpec(
        "tx_frequency_deviation",
        "account_transactions",
        TEMPORAL_WINDOWS["1h"],
        "rolling_count_zscore",
        0.0,
    ),
    "geo_distance_km": FeatureSpec(
        "geo_distance_km", "account_transactions", None, "previous_distance", 0.0
    ),
    "geo_speed_kmh": FeatureSpec(
        "geo_speed_kmh", "account_transactions", None, "previous_speed", 0.0
    ),
    "impossible_travel": FeatureSpec(
        "impossible_travel", "account_transactions", None, "threshold", 0.0
    ),
    "account_age_days": FeatureSpec("account_age_days", "account", None, "age", 0.0),
    "cold_start": FeatureSpec("cold_start", "account_transactions", None, "indicator", 1.0),
}


FEATURE_COLUMNS: tuple[str, ...] = tuple(FEATURE_REGISTRY)
