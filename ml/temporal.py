"""Point-in-time temporal burst features.

For transaction ``T``, the current rolling count and volume are measured over
``[T - window, T)``.  Their baseline is the distribution of the same rolling
measure at earlier account transaction timestamps.  Each historical anchor is
queried through ``TransactionHistory.as_of(anchor)``, so neither the anchored row,
equal-timestamp peers, nor future rows can affect a score.

Checkpoint 2C-P note: this module has two implementations of the same
semantics. :func:`temporal_features` is the original, per-transaction
reference: correct and simple, but ``O(k)`` per call in the number ``k`` of an
account's earlier distinct timestamps, because it reconstructs every earlier
anchor's rolling aggregate from scratch. Called once per transaction over an
account's full history, that is quadratic overall, which was the measured
Checkpoint 2C runtime bottleneck. :func:`compute_temporal_features` computes
the identical values in a single chronological pass per account using
incremental sliding windows and Welford's online mean/variance, and is what
``ml.features.build_feature_matrix`` actually calls.
:func:`temporal_features` is kept unchanged and is the correctness oracle
that ``tests/leakage/test_temporal_bulk_equivalence.py`` checks the bulk path
against; it is also still exercised directly by
``tests/leakage/test_temporal.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
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

    This is the reference implementation (see module docstring): correct, but
    ``O(k)`` per call. Kept unchanged as the correctness oracle for
    :func:`compute_temporal_features`.
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


def compute_temporal_features(
    history_transactions: Iterable[Any], targets: Sequence[Any]
) -> dict[Any, dict[str, float]]:
    """Bulk, chronological equivalent of calling :func:`temporal_features` once
    per target transaction.

    Why this is equivalent (not just similar): the "anchor" value
    ``temporal_features`` reconstructs for an earlier account timestamp ``t_a``
    -- the rolling count/volume in ``[t_a - window, t_a)`` -- is *exactly* the
    same number as the "current" rolling count/volume that would be computed
    if ``t_a``'s own row(s) were themselves the target. So each anchor needs
    computing only once, at the moment its own group is visited, and can then
    be folded into a running baseline statistic for every later group. Walking
    one account's rows in chronological order, that is a single linear pass
    per window per account: a two-pointer sliding count/sum for the "current"
    value, and Welford's online algorithm for the baseline mean/variance,
    both O(1) amortized per step, instead of an O(k) reconstruction per
    target.

    Returns the six temporal features per transaction, keyed by
    ``transaction_id``, for every row in ``targets``. Requires every target to
    also be present in ``history_transactions`` -- true of every current
    caller (``build_feature_matrix`` always folds its targets into the history
    it builds). Any target not found there falls back to the reference
    :func:`temporal_features` for that one row, so correctness never depends
    on the fast path's assumption holding.
    """
    history_list = list(history_transactions)

    by_account: dict[Any, list[tuple[datetime, float, Any]]] = {}
    for row in history_list:
        account_id = _value(row, "account_id")
        by_account.setdefault(account_id, []).append(
            (
                _value(row, "timestamp"),
                float(_value(row, "amount")),
                _value(row, "transaction_id"),
            )
        )

    results: dict[Any, dict[str, float]] = {}

    for rows in by_account.values():
        rows.sort(key=lambda item: (item[0], item[2]))
        n = len(rows)
        timestamps = [item[0] for item in rows]
        amounts = [item[1] for item in rows]
        tx_ids = [item[2] for item in rows]

        # Distinct-timestamp group boundaries. Rows sharing an exact
        # timestamp collapse into one anchor, matching temporal_features'
        # `set(row["timestamp"] for row in prior_rows)` dedup.
        group_starts = [0]
        for i in range(1, n):
            if timestamps[i] != timestamps[i - 1]:
                group_starts.append(i)
        group_starts.append(n)

        # Independent incremental state per window: a two-pointer sliding
        # count/sum over this account's own rows (`left`/`right`/`sum`), plus
        # a Welford accumulator (`n_b`/`mean_*`/`m2_*`) for the baseline
        # distribution of that rolling value sampled at each earlier
        # distinct account timestamp.
        state = {
            label: {
                "left": 0,
                "right": 0,
                "sum": 0.0,
                "n_b": 0,
                "mean_c": 0.0,
                "m2_c": 0.0,
                "mean_v": 0.0,
                "m2_v": 0.0,
            }
            for label in TEMPORAL_WINDOWS
        }

        for group_index in range(len(group_starts) - 1):
            g_start, g_end = group_starts[group_index], group_starts[group_index + 1]
            t_g = timestamps[g_start]
            group_feature: dict[str, float] = {}

            for label, window in TEMPORAL_WINDOWS.items():
                st = state[label]
                lower = t_g - window
                while st["left"] < st["right"] and timestamps[st["left"]] < lower:
                    st["sum"] -= amounts[st["left"]]
                    st["left"] += 1

                current_count = float(st["right"] - st["left"])
                current_volume = float(st["sum"])

                if st["n_b"] < MIN_HISTORY_FOR_DEVIATION:
                    z_count = 0.0
                    z_volume = 0.0
                else:
                    var_c = st["m2_c"] / (st["n_b"] - 1)
                    std_c = math.sqrt(max(0.0, var_c))
                    if std_c <= MIN_STD:
                        z_count = 0.0
                    else:
                        z_count = (current_count - st["mean_c"]) / std_c
                        z_count = z_count if math.isfinite(z_count) else 0.0

                    var_v = st["m2_v"] / (st["n_b"] - 1)
                    std_v = math.sqrt(max(0.0, var_v))
                    if std_v <= MIN_STD:
                        z_volume = 0.0
                    else:
                        z_volume = (current_volume - st["mean_v"]) / std_v
                        z_volume = z_volume if math.isfinite(z_volume) else 0.0

                group_feature[f"temporal_zscore_count_{label}"] = float(z_count)
                group_feature[f"temporal_zscore_volume_{label}"] = float(z_volume)

                # Fold this group's own (count, volume) into the baseline
                # *after* using it to score the group -- an anchor must never
                # include itself -- then extend the window to include the
                # group's own rows so later groups can see them.
                st["n_b"] += 1
                delta_c = current_count - st["mean_c"]
                st["mean_c"] += delta_c / st["n_b"]
                st["m2_c"] += delta_c * (current_count - st["mean_c"])
                delta_v = current_volume - st["mean_v"]
                st["mean_v"] += delta_v / st["n_b"]
                st["m2_v"] += delta_v * (current_volume - st["mean_v"])

                for idx in range(g_start, g_end):
                    st["sum"] += amounts[idx]
                st["right"] = g_end

            for idx in range(g_start, g_end):
                results[tx_ids[idx]] = group_feature

    fallback_history: TransactionHistory | None = None
    output: dict[Any, dict[str, float]] = {}
    for target in targets:
        tx_id = _value(target, "transaction_id")
        if tx_id in results:
            output[tx_id] = results[tx_id]
        else:
            if fallback_history is None:
                fallback_history = TransactionHistory.from_transactions(history_list)
            output[tx_id] = temporal_features(fallback_history, target)
    return output
