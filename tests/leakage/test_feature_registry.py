"""The declared feature catalogue must exactly match extractor output."""

from __future__ import annotations

from datetime import timedelta

import pytest

from ml.feature_registry import FEATURE_COLUMNS, FEATURE_REGISTRY
from ml.features import build_feature_matrix

pytestmark = pytest.mark.leakage


def test_registry_and_output_columns_are_exactly_aligned(make_txn, origin):
    accounts = {"acc_1": {"created_at": origin - timedelta(days=10)}}
    target = make_txn(99, minutes=60)
    history = [make_txn(index, minutes=index * 10) for index in range(6)]
    matrix = build_feature_matrix(
        [target], history_transactions=[*history, target], accounts=accounts
    )

    assert tuple(matrix.columns) == FEATURE_COLUMNS
    assert set(matrix.columns) == set(FEATURE_REGISTRY)
    assert len(FEATURE_COLUMNS) == len(FEATURE_REGISTRY)
    assert all(spec.point_in_time_safe for spec in FEATURE_REGISTRY.values())
    assert matrix.loc[0, "tx_frequency_deviation"] == matrix.loc[
        0, "temporal_zscore_count_1h"
    ]
