"""Regression test for the Checkpoint 2H feature-column-ordering bug.

``scripts/evaluate_heldout.py``'s ``_apply_frozen_scaling()`` reconstructs the
StandardScaler mean/scale from a persisted JSON file written with
``json.dumps(..., sort_keys=True)`` (``scripts/train_baseline_models.py``),
which alphabetically reorders the mean/scale sub-dictionaries' keys on disk.
These tests prove the reconstructed, scaled matrix's columns are always the
canonical ``FEATURE_COLUMNS`` order regardless of what key order the on-disk
JSON happens to use, that the arithmetic itself is still correct, and that
``build_feature_matrix`` gives identical canonical order for independently
built row sets (standing in for different splits). No heldout data is read
anywhere in this file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ml.feature_registry import FEATURE_COLUMNS
from ml.features import build_feature_matrix
from scripts.evaluate_heldout import _apply_frozen_scaling
from simulator.transactions import Transaction

pytestmark = pytest.mark.ml


def _base_row(**overrides: float) -> dict[str, float]:
    row = {column: 1.0 for column in FEATURE_COLUMNS}
    row.update(overrides)
    return row


def _toy_matrix(rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame[list(FEATURE_COLUMNS)]


def test_apply_frozen_scaling_returns_canonical_order_from_alphabetical_json_keys():
    matrix = _toy_matrix(
        [_base_row(transaction_amount=10.0), _base_row(transaction_amount=20.0)]
    )

    # Exactly what json.dumps(..., sort_keys=True) does on disk: the mean/
    # scale dict keys end up alphabetically sorted, not in FEATURE_COLUMNS
    # order -- a non-coincidental, worst-case ordering to reconstruct from.
    alphabetical_order = sorted(FEATURE_COLUMNS)
    scaler_params = {
        "mean": {column: 0.0 for column in alphabetical_order},
        "scale": {column: 1.0 for column in alphabetical_order},
    }

    scaled = _apply_frozen_scaling(matrix, scaler_params)

    assert list(scaled.columns) == list(FEATURE_COLUMNS)


def test_apply_frozen_scaling_is_numerically_correct():
    matrix = _toy_matrix([_base_row(transaction_amount=10.0)])
    scaler_params = {
        "mean": {
            column: (5.0 if column == "transaction_amount" else 0.0)
            for column in sorted(FEATURE_COLUMNS)
        },
        "scale": {
            column: (2.0 if column == "transaction_amount" else 1.0)
            for column in sorted(FEATURE_COLUMNS)
        },
    }

    scaled = _apply_frozen_scaling(matrix, scaler_params)

    assert scaled.loc[0, "transaction_amount"] == pytest.approx((10.0 - 5.0) / 2.0)
    # An untouched column: mean 0.0, scale 1.0 -> value unchanged.
    assert scaled.loc[0, "cold_start"] == pytest.approx(1.0)


def _toy_transaction(index: int, minutes: float, amount: float) -> Transaction:
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Transaction(
        transaction_id=f"txn_{index:05d}",
        timestamp=origin + timedelta(minutes=minutes),
        account_id="acc_1",
        customer_id="cus_1",
        merchant_id="mer_1",
        beneficiary_id=None,
        device_id="dev_1",
        ip_id="ip_1",
        amount=amount,
        currency="USD",
        channel="web",
        transaction_type="purchase",
        lat=10.0,
        lon=20.0,
        status="completed",
        is_fraud=False,
        attack_id=None,
    )


def test_build_feature_matrix_uses_identical_canonical_order_across_row_sets():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    accounts = {"acc_1": {"created_at": origin - timedelta(days=10)}}

    set_a = [_toy_transaction(i, i * 5.0, 10.0 + i) for i in range(5)]
    set_b = [_toy_transaction(100 + i, i * 7.0, 50.0 + i) for i in range(5)]

    matrix_a = build_feature_matrix(set_a, history_transactions=set_a, accounts=accounts)
    matrix_b = build_feature_matrix(set_b, history_transactions=set_b, accounts=accounts)

    assert list(matrix_a.columns) == list(FEATURE_COLUMNS)
    assert list(matrix_b.columns) == list(FEATURE_COLUMNS)
    assert list(matrix_a.columns) == list(matrix_b.columns)
