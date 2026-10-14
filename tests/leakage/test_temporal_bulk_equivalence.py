"""Checkpoint 2C-P equivalence and leakage tests for the bulk temporal path.

Compares ``compute_temporal_features`` (the chronological/incremental
implementation used by ``build_feature_matrix``) against ``temporal_features``
(the unchanged, O(k)-per-call reference) across the scenarios enumerated in
the Checkpoint 2C-P task: normal history, cold start, insufficient history,
exact window boundaries, repeated timestamps, accounts with many
transactions, and interleaved multi-account history. Also re-verifies,
directly against the bulk path, the leakage properties already required of
the reference implementation.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from ml.features import build_feature_matrix
from ml.point_in_time import TransactionHistory
from ml.temporal import compute_temporal_features, temporal_features

pytestmark = pytest.mark.leakage

TEMPORAL_KEYS = (
    "temporal_zscore_count_15m",
    "temporal_zscore_count_1h",
    "temporal_zscore_count_6h",
    "temporal_zscore_volume_15m",
    "temporal_zscore_volume_1h",
    "temporal_zscore_volume_6h",
)

# Tolerance justification: the bulk path sums window volumes via a running
# add/subtract (O(1) amortized) instead of the reference's fresh sum() per
# query, and folds baseline mean/variance with Welford's online algorithm
# instead of a two-pass math.fsum. Both are standard, numerically stable
# reformulations of the same arithmetic; for float64 with the amount
# magnitudes and window/account sizes used here, the two accumulation
# strategies are expected to agree to within a few ULPs -- many orders of
# magnitude tighter than this bound. 1e-6 (absolute or relative) is chosen to
# sit comfortably above realistic float64 drift while remaining tight enough
# to catch an actual algorithmic mismatch.
TOLERANCE = 1e-6


def _assert_equivalent(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual) == set(expected) == set(TEMPORAL_KEYS)
    for key in TEMPORAL_KEYS:
        assert math.isclose(actual[key], expected[key], rel_tol=TOLERANCE, abs_tol=TOLERANCE), (
            f"{key}: bulk={actual[key]!r} reference={expected[key]!r}"
        )


def _reference_results(history_transactions, targets):
    history = TransactionHistory.from_transactions(history_transactions)
    return {target.transaction_id: temporal_features(history, target) for target in targets}


def _check_scenario(history_transactions, targets) -> None:
    expected = _reference_results(history_transactions, targets)
    actual = compute_temporal_features(history_transactions, targets)
    assert set(actual) == set(expected)
    for tx_id in expected:
        _assert_equivalent(actual[tx_id], expected[tx_id])


# ---------------------------------------------------------------------------
# Equivalence: bulk vs reference
# ---------------------------------------------------------------------------


def test_equivalence_normal_history(make_txn):
    """A normal account with a modest, irregular transaction history."""
    txns = [
        make_txn(i, minutes=m, amount=a)
        for i, (m, a) in enumerate(
            [(0, 50), (5, 75), (12, 20), (40, 500), (65, 30), (90, 45), (130, 900), (200, 60)]
        )
    ]
    _check_scenario(txns, txns)


def test_equivalence_first_transaction_cold_start(make_txn):
    """The very first transaction for an account: zero history."""
    target = make_txn(0, minutes=0, amount=100.0)
    _check_scenario([target], [target])


def test_equivalence_insufficient_history(make_txn):
    """Fewer than MIN_HISTORY_FOR_DEVIATION (5) prior distinct timestamps."""
    txns = [make_txn(i, minutes=i * 10, amount=100.0 + i) for i in range(4)]
    target = make_txn(99, minutes=90, amount=250.0)
    _check_scenario([*txns, target], [*txns, target])


def test_equivalence_exact_window_boundaries(make_txn):
    """Rows planted exactly on each window's lower edge and just outside it."""
    earlier = [make_txn(i, minutes=i * 5, amount=10.0 + i) for i in range(6)]
    on_15m_edge = make_txn(50, minutes=105, amount=111.0)  # target - 15m exactly
    outside_15m = make_txn(51, minutes=104, amount=222.0)  # target - 15m - 1min
    on_1h_edge = make_txn(52, minutes=60, amount=333.0)
    on_6h_edge = make_txn(53, minutes=0, amount=444.0)
    target = make_txn(99, minutes=120, amount=500.0)
    txns = [*earlier, on_15m_edge, outside_15m, on_1h_edge, on_6h_edge, target]
    _check_scenario(txns, txns)


def test_equivalence_repeated_timestamps(make_txn):
    """Several transactions sharing exactly the same timestamp."""
    tied_a = make_txn(1, minutes=30, amount=40.0)
    tied_b = make_txn(2, minutes=30, amount=60.0)
    tied_c = make_txn(3, minutes=30, amount=80.0)
    filler = [make_txn(10 + i, minutes=30 + i, amount=15.0 + i) for i in range(5)]
    target = make_txn(99, minutes=90, amount=200.0)
    txns = [tied_a, tied_b, tied_c, *filler, target]
    _check_scenario(txns, txns)


def test_equivalence_many_transactions_one_account(make_txn):
    """An account with many transactions, exercising the incremental state
    across a long chronological run -- the shape of the Checkpoint 2C
    bottleneck."""
    txns = [make_txn(i, minutes=i * 3.0, amount=20.0 + (i % 17) * 5.5) for i in range(400)]
    _check_scenario(txns, txns)


def test_equivalence_multiple_accounts_interleaved(make_txn):
    """Two accounts interleaved in time; each account's baseline must only
    see its own history, never the other account's."""
    txns = []
    for i in range(60):
        account = "acc_1" if i % 2 == 0 else "acc_2"
        txns.append(make_txn(i, minutes=i * 4.0, amount=25.0 + i, account=account))
    _check_scenario(txns, txns)


# ---------------------------------------------------------------------------
# Leakage properties, re-verified directly against the bulk path
# ---------------------------------------------------------------------------


def _accounts(origin):
    return {"acc_1": {"created_at": origin - timedelta(days=10)}}


def _temporal_columns(matrix):
    return [column for column in matrix.columns if column.startswith("temporal_")]


def test_bulk_same_timestamp_rows_do_not_affect_result(make_txn, origin):
    prior = [make_txn(i, minutes=i * 10, amount=100.0 + i) for i in range(6)]
    target = make_txn(99, minutes=90, amount=200.0)
    peer = make_txn(100, minutes=90, amount=1_000_000.0)

    baseline = build_feature_matrix(
        [target], history_transactions=[*prior, target], accounts=_accounts(origin)
    )
    with_peer = build_feature_matrix(
        [target], history_transactions=[*prior, target, peer], accounts=_accounts(origin)
    )
    assert baseline[_temporal_columns(baseline)].equals(with_peer[_temporal_columns(with_peer)])


def test_bulk_future_rows_do_not_affect_result(make_txn, origin):
    prior = [make_txn(i, minutes=i * 10, amount=100.0 + i) for i in range(6)]
    target = make_txn(99, minutes=90, amount=200.0)
    future = [make_txn(200 + i, minutes=120 + i, amount=1_000_000.0) for i in range(8)]

    baseline = build_feature_matrix(
        [target], history_transactions=[*prior, target], accounts=_accounts(origin)
    )
    with_future = build_feature_matrix(
        [target], history_transactions=[*prior, target, *future], accounts=_accounts(origin)
    )
    assert baseline[_temporal_columns(baseline)].equals(with_future[_temporal_columns(with_future)])


def test_bulk_cold_start_is_zero_and_finite(make_txn):
    target = make_txn(99, minutes=60, amount=200.0)
    features = compute_temporal_features([target], [target])[target.transaction_id]

    assert set(features) == set(TEMPORAL_KEYS)
    assert all(value == 0.0 for value in features.values())
    assert np.isfinite(list(features.values())).all()


def test_bulk_window_includes_lower_edge_and_excludes_upper_edge(make_txn):
    target = make_txn(99, minutes=120, amount=100.0)
    earlier = [make_txn(10 + i, minutes=i * 10, amount=10.0 + i) for i in range(6)]

    lower_edge = make_txn(1, minutes=60, amount=500.0)
    with_edge = compute_temporal_features([*earlier, lower_edge, target], [target])[
        target.transaction_id
    ]

    outside = make_txn(2, minutes=59, amount=500.0)
    without_edge = compute_temporal_features([*earlier, outside, target], [target])[
        target.transaction_id
    ]

    assert with_edge["temporal_zscore_volume_1h"] != without_edge["temporal_zscore_volume_1h"]


def test_bulk_features_do_not_depend_on_labels_or_attack_metadata(make_txn):
    normal = [make_txn(i, minutes=i * 10, amount=100.0 + i) for i in range(6)]
    labelled = [make_txn(i, minutes=i * 10, amount=100.0 + i, is_fraud=True) for i in range(6)]
    target_normal = make_txn(99, minutes=90, amount=200.0)
    target_labelled = make_txn(99, minutes=90, amount=200.0, is_fraud=True)

    clean = compute_temporal_features([*normal, target_normal], [target_normal])[
        target_normal.transaction_id
    ]
    labelled_result = compute_temporal_features(
        [*labelled, target_labelled], [target_labelled]
    )[target_labelled.transaction_id]

    assert clean == labelled_result
