"""Checkpoint 2D: preprocessing / model-matrix layer tests.

Covers the properties this checkpoint exists to guarantee: preprocessing
parameters are fit on TRAIN only, validation/held-out-shaped data never
influences them, transformation is deterministic, the 25-feature schema is
preserved exactly, and no label, attack-metadata, or identifier column can
reach the model matrix.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from ml.feature_registry import FEATURE_COLUMNS
from ml.features import build_feature_matrix
from ml.preprocessing import StandardScaler, assert_model_matrix_safe
from ml.split_policy import SplitPolicyError
from simulator.leakage import LeakageError

pytestmark = pytest.mark.leakage


def _accounts(origin):
    return {"acc_1": {"created_at": origin - timedelta(days=10)}}


def _real_feature_matrix(make_txn, origin, n=20):
    txns = [make_txn(i, minutes=i * 7.0, amount=20.0 + (i % 5) * 15.5) for i in range(n)]
    return build_feature_matrix(txns, history_transactions=txns, accounts=_accounts(origin))


def _base_row(**overrides: float) -> dict[str, float]:
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row.update(overrides)
    return row


def _toy_matrix(rows: list[dict[str, float]]) -> pd.DataFrame:
    """A hand-built matrix over the real 25-column schema, for exact arithmetic
    checks. Every registered column is present so it passes
    ``assert_model_matrix_safe``; values are otherwise arbitrary and PIT-
    irrelevant since this module operates purely on an already-built feature
    matrix, not on feature extraction."""
    frame = pd.DataFrame(rows)
    return frame[list(FEATURE_COLUMNS)]


# ---------------------------------------------------------------------------
# Fit on train only
# ---------------------------------------------------------------------------


def test_fit_computes_mean_and_std_from_train_data_only():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (10.0, 20.0, 30.0)])
    scaler = StandardScaler().fit(train, split="train")

    expected_mean = train["transaction_amount"].mean()
    expected_std = train["transaction_amount"].std(ddof=0)
    assert scaler.mean_["transaction_amount"] == pytest.approx(expected_mean)
    assert scaler.scale_["transaction_amount"] == pytest.approx(expected_std)


@pytest.mark.parametrize("split", ["validation", "heldout", "warmup"])
def test_fit_is_refused_on_every_split_but_train(split):
    matrix = _toy_matrix([_base_row(transaction_amount=v) for v in (1.0, 2.0, 3.0)])
    scaler = StandardScaler()
    with pytest.raises(SplitPolicyError):
        scaler.fit(matrix, split=split)
    assert not scaler.is_fitted


def test_refitting_is_refused_and_leaves_original_parameters_untouched():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (1.0, 2.0, 3.0)])
    scaler = StandardScaler().fit(train, split="train")
    other_train = _toy_matrix([_base_row(transaction_amount=v) for v in (100.0, 200.0)])
    with pytest.raises(SplitPolicyError):
        scaler.fit(other_train, split="train")
    assert scaler.mean_["transaction_amount"] == pytest.approx(2.0)


def test_transform_before_fit_is_refused():
    matrix = _toy_matrix([_base_row(transaction_amount=1.0)])
    with pytest.raises(SplitPolicyError):
        StandardScaler().transform(matrix, split="validation")


# ---------------------------------------------------------------------------
# Validation/held-out must never influence fitted parameters
# ---------------------------------------------------------------------------


def test_validation_shaped_data_does_not_influence_fitted_parameters():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (10.0, 20.0, 30.0)])
    scaler = StandardScaler().fit(train, split="train")
    frozen_mean = dict(scaler.mean_)
    frozen_scale = dict(scaler.scale_)

    validation_a = _toy_matrix([_base_row(transaction_amount=v) for v in (1000.0, 2000.0)])
    validation_b = _toy_matrix([_base_row(transaction_amount=v) for v in (-500.0, 0.0, 500.0)])
    scaler.transform(validation_a, split="validation")
    scaler.transform(validation_b, split="validation")

    assert scaler.mean_ == frozen_mean
    assert scaler.scale_ == frozen_scale


def test_validation_is_transformed_with_train_parameters_not_its_own():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (10.0, 20.0, 30.0)])
    scaler = StandardScaler().fit(train, split="train")

    validation = _toy_matrix([_base_row(transaction_amount=v) for v in (100.0, 200.0)])
    transformed = scaler.transform(validation, split="validation")

    train_mean = train["transaction_amount"].mean()
    train_std = train["transaction_amount"].std(ddof=0)
    expected = (validation["transaction_amount"] - train_mean) / train_std
    assert transformed["transaction_amount"].to_numpy() == pytest.approx(expected.to_numpy())


def test_heldout_shaped_data_cannot_be_fitted_on_either():
    matrix = _toy_matrix([_base_row(transaction_amount=v) for v in (1.0, 2.0, 3.0)])
    with pytest.raises(SplitPolicyError):
        StandardScaler().fit(matrix, split="heldout")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_transform_is_deterministic():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (5.0, 15.0, 25.0, 35.0)])
    scaler = StandardScaler().fit(train, split="train")
    target = _toy_matrix([_base_row(transaction_amount=v) for v in (7.0, 42.0)])

    first = scaler.transform(target, split="validation")
    second = scaler.transform(target, split="validation")
    assert first.equals(second)


def test_two_independently_fitted_scalers_on_identical_train_data_agree():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (5.0, 15.0, 25.0)])
    a = StandardScaler().fit(train, split="train")
    b = StandardScaler().fit(train.copy(), split="train")
    assert a.mean_ == b.mean_
    assert a.scale_ == b.scale_


def test_zero_variance_train_column_is_centered_not_divided_by_zero():
    train = _toy_matrix([_base_row(cold_start=1.0) for _ in range(5)])
    scaler = StandardScaler().fit(train, split="train")
    assert scaler.scale_["cold_start"] == pytest.approx(1.0)
    assert scaler.mean_["cold_start"] == pytest.approx(1.0)

    target = _toy_matrix([_base_row(cold_start=1.0)])
    transformed = scaler.transform(target, split="validation")
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()
    assert transformed["cold_start"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Schema preservation
# ---------------------------------------------------------------------------


def test_output_preserves_the_25_feature_columns_in_order():
    train = _toy_matrix([_base_row(transaction_amount=v) for v in (1.0, 2.0, 3.0)])
    scaler = StandardScaler().fit(train, split="train")
    transformed = scaler.transform(train, split="train")
    assert list(transformed.columns) == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 25


def test_assert_model_matrix_safe_accepts_the_real_feature_matrix(make_txn, origin):
    matrix = _real_feature_matrix(make_txn, origin)
    assert_model_matrix_safe(matrix)  # must not raise


def test_assert_model_matrix_safe_rejects_missing_or_extra_columns():
    incomplete = _toy_matrix([_base_row()]).drop(columns=["transaction_amount"])
    with pytest.raises(ValueError):
        assert_model_matrix_safe(incomplete)

    extra = _toy_matrix([_base_row()])
    extra["unexpected_column"] = 1.0
    with pytest.raises(ValueError):
        assert_model_matrix_safe(extra)


def test_assert_model_matrix_safe_rejects_non_finite_values():
    bad_nan = _toy_matrix([_base_row(transaction_amount=float("nan"))])
    with pytest.raises(ValueError):
        assert_model_matrix_safe(bad_nan)

    bad_inf = _toy_matrix([_base_row(transaction_amount=float("inf"))])
    with pytest.raises(ValueError):
        assert_model_matrix_safe(bad_inf)


# ---------------------------------------------------------------------------
# No labels / attack metadata / identifiers reach the model matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_column", ["is_fraud", "attack_id", "transaction_id", "account_id", "customer_id"]
)
def test_assert_model_matrix_safe_rejects_labels_and_identifiers(forbidden_column):
    tainted = _toy_matrix([_base_row()])
    tainted[forbidden_column] = 0
    with pytest.raises(LeakageError):
        assert_model_matrix_safe(tainted)


def test_fit_input_is_also_checked_for_forbidden_columns():
    tainted = _toy_matrix([_base_row()])
    tainted["is_fraud"] = False
    with pytest.raises(LeakageError):
        StandardScaler().fit(tainted, split="train")


def test_real_feature_matrix_round_trips_through_the_scaler(make_txn, origin):
    matrix = _real_feature_matrix(make_txn, origin, n=30)
    scaler = StandardScaler().fit(matrix, split="train")
    transformed = scaler.transform(matrix, split="train")

    assert list(transformed.columns) == list(FEATURE_COLUMNS)
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()
    assert_model_matrix_safe(transformed)
