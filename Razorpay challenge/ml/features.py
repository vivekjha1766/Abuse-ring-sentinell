"""Point-in-time-safe, transaction-level feature extraction.

Only data observable at transaction authorization is used directly. Historical
signals are read exclusively through :class:`ml.point_in_time.TransactionHistory`,
whose ``ts < T`` cut also excludes same-timestamp rows.  Labels and attack metadata
never enter the returned matrix.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.feature_contract import assert_model_input_safe
from ml.feature_registry import FEATURE_COLUMNS, WINDOWS
from ml.point_in_time import MIN_HISTORY_FOR_DEVIATION, TransactionHistory
from ml.temporal import temporal_features

IMPOSSIBLE_TRAVEL_MAX_KMH = 900.0


def _value(row: Any, field: str) -> Any:
    return row[field] if isinstance(row, Mapping) else getattr(row, field)


def _account_created_at(accounts: Mapping[str, Any], account_id: str) -> datetime:
    account = accounts[account_id]
    return _value(account, "created_at")


def build_feature_matrix(
    transactions: Sequence[Any],
    *,
    history_transactions: Iterable[Any],
    accounts: Mapping[str, Any],
) -> pd.DataFrame:
    """Return model-safe features for ``transactions`` as of each row's timestamp.

    ``history_transactions`` may contain warm-up, train, validation, and the target
    rows themselves.  This is safe because each per-row view is strictly earlier
    than that row's timestamp; callers must still respect split policy and never
    pass held-out rows before Checkpoint 6.
    """
    history = TransactionHistory.from_transactions(history_transactions)
    records: list[dict[str, float]] = []

    for txn in transactions:
        timestamp = _value(txn, "timestamp")
        account_id = _value(txn, "account_id")
        amount = float(_value(txn, "amount"))
        view = history.as_of(timestamp)
        prior_count = view.count(key="account_id", value=account_id)
        previous = view.previous(key="account_id", value=account_id)
        distance = view.distance_from_km(
            float(_value(txn, "lat")), float(_value(txn, "lon")),
            key="account_id", value=account_id,
        )

        speed = 0.0
        if previous is not None and distance is not None:
            elapsed_hours = (timestamp - previous["timestamp"]).total_seconds() / 3600.0
            if elapsed_hours > 0.0:
                speed = float(distance / elapsed_hours)

        record: dict[str, float] = {
            "transaction_amount": amount,
            "log_transaction_amount": float(math.log1p(max(amount, 0.0))),
            "amount_deviation": float(view.deviation(amount, key="account_id", value=account_id)),
            "geo_distance_km": float(distance or 0.0),
            "geo_speed_kmh": speed,
            "impossible_travel": float(speed > IMPOSSIBLE_TRAVEL_MAX_KMH),
            "account_age_days": float(view.age_days(_account_created_at(accounts, account_id))),
            "cold_start": float(prior_count < MIN_HISTORY_FOR_DEVIATION),
        }
        for label, window in WINDOWS.items():
            record[f"tx_velocity_{label}"] = float(
                view.count(window, key="account_id", value=account_id)
            )
            record[f"tx_volume_{label}"] = float(
                view.amount_sum(window, key="account_id", value=account_id)
            )
        temporal = temporal_features(history, txn)
        record.update(temporal)
        # The catalogue's behavioural frequency-deviation feature is the
        # historical z-score of the current one-hour account count. It is kept
        # under its behavioural name as well as in the temporal feature group.
        record["tx_frequency_deviation"] = temporal["temporal_zscore_count_1h"]
        records.append(record)

    matrix = pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
    if not matrix.empty and not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("feature extraction produced NaN or infinity")
    assert_model_input_safe(matrix.columns.tolist(), context="extracted feature matrix")
    return matrix
