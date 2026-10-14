"""Feature-engine tests for PIT safety and label quarantine."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from ml.features import build_feature_matrix

pytestmark = pytest.mark.leakage


def _accounts(origin):
    return {"acc_1": {"created_at": origin - timedelta(days=10)}}


def test_features_exclude_same_timestamp_and_future_rows(make_txn, origin):
    earlier = make_txn(1, minutes=30, amount=100.0)
    target = make_txn(2, minutes=60, amount=200.0)
    simultaneous = make_txn(3, minutes=60, amount=9999.0)
    future = make_txn(4, minutes=120, amount=9999.0)

    baseline = build_feature_matrix([target], history_transactions=[earlier, target], accounts=_accounts(origin))
    contaminated = build_feature_matrix(
        [target], history_transactions=[earlier, target, simultaneous, future], accounts=_accounts(origin)
    )

    assert baseline.equals(contaminated)
    assert baseline.loc[0, "tx_velocity_1h"] == 1.0
    assert baseline.loc[0, "tx_volume_1h"] == 100.0


def test_features_are_finite_and_do_not_emit_forbidden_fields(make_txn, origin):
    target = make_txn(1, minutes=0, amount=50.0, is_fraud=True)
    matrix = build_feature_matrix([target], history_transactions=[target], accounts=_accounts(origin))

    assert np.isfinite(matrix.to_numpy(dtype=float)).all()
    assert {"is_fraud", "attack_id", "transaction_id", "account_id"}.isdisjoint(matrix.columns)
    assert matrix.loc[0, "cold_start"] == 1.0
