"""Temporal-feature leakage tests: strict history, not transaction order."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from ml.features import build_feature_matrix
from ml.temporal import temporal_features
from ml.point_in_time import TransactionHistory

pytestmark = pytest.mark.leakage


def _accounts(origin):
    return {"acc_1": {"created_at": origin - timedelta(days=10)}}


def _temporal_columns(matrix):
    return [column for column in matrix.columns if column.startswith("temporal_")]


def test_temporal_features_exclude_same_timestamp_rows(make_txn, origin):
    prior = [make_txn(index, minutes=index * 10, amount=100.0 + index) for index in range(6)]
    target = make_txn(99, minutes=90, amount=200.0)
    peer = make_txn(100, minutes=90, amount=1_000_000.0)

    baseline = build_feature_matrix([target], history_transactions=[*prior, target], accounts=_accounts(origin))
    with_peer = build_feature_matrix(
        [target], history_transactions=[*prior, target, peer], accounts=_accounts(origin)
    )

    assert baseline[_temporal_columns(baseline)].equals(with_peer[_temporal_columns(with_peer)])


def test_temporal_features_are_invariant_to_future_rows(make_txn, origin):
    prior = [make_txn(index, minutes=index * 10, amount=100.0 + index) for index in range(6)]
    target = make_txn(99, minutes=90, amount=200.0)
    future = [make_txn(200 + index, minutes=120 + index, amount=1_000_000.0) for index in range(8)]

    baseline = build_feature_matrix([target], history_transactions=[*prior, target], accounts=_accounts(origin))
    with_future = build_feature_matrix(
        [target], history_transactions=[*prior, target, *future], accounts=_accounts(origin)
    )

    assert baseline[_temporal_columns(baseline)].equals(with_future[_temporal_columns(with_future)])


def test_temporal_cold_start_is_zero_and_finite(make_txn):
    target = make_txn(99, minutes=60, amount=200.0)
    features = temporal_features(TransactionHistory.from_transactions([]), target)

    assert len(features) == 6
    assert set(features) == {
        "temporal_zscore_count_15m", "temporal_zscore_count_1h", "temporal_zscore_count_6h",
        "temporal_zscore_volume_15m", "temporal_zscore_volume_1h", "temporal_zscore_volume_6h",
    }
    assert all(value == 0.0 for value in features.values())
    assert np.isfinite(list(features.values())).all()


def test_temporal_window_includes_lower_edge_and_excludes_upper_edge(make_txn):
    target = make_txn(99, minutes=120, amount=100.0)
    lower_edge = make_txn(1, minutes=60, amount=500.0)
    outside = make_txn(2, minutes=59, amount=500.0)
    earlier = [make_txn(10 + index, minutes=index * 10, amount=10.0 + index) for index in range(6)]

    with_edge = temporal_features(TransactionHistory.from_transactions([*earlier, lower_edge]), target)
    without_edge = temporal_features(TransactionHistory.from_transactions([*earlier, outside]), target)

    assert with_edge["temporal_zscore_volume_1h"] != without_edge["temporal_zscore_volume_1h"]


def test_temporal_features_do_not_depend_on_labels_or_attack_metadata(make_txn):
    normal = [make_txn(index, minutes=index * 10, amount=100.0 + index) for index in range(6)]
    labelled = [make_txn(index, minutes=index * 10, amount=100.0 + index, is_fraud=True) for index in range(6)]
    target_normal = make_txn(99, minutes=90, amount=200.0)
    target_labelled = make_txn(99, minutes=90, amount=200.0, is_fraud=True)

    clean = temporal_features(TransactionHistory.from_transactions(normal), target_normal)
    labelled_result = temporal_features(TransactionHistory.from_transactions(labelled), target_labelled)

    assert clean == labelled_result
