"""Checkpoint 2E: split-policy and no-leakage tests for ml/models.py.

Covers: Model 0's predictions are provably independent of feature values;
fitting either model is refused on every split but train; refitting is
refused; predicting before fit is refused; no label/identifier column can
reach either model's fit or predict_proba; logistic regression uses the
documented class-weighting approach and is deterministic given a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.feature_registry import FEATURE_COLUMNS
from ml.models import LogisticRegressionBaseline, MajorityClassifier
from ml.split_policy import SplitPolicyError
from simulator.leakage import LeakageError

pytestmark = pytest.mark.leakage


def _base_row(**overrides: float) -> dict[str, float]:
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row.update(overrides)
    return row


def _toy_matrix(rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame[list(FEATURE_COLUMNS)]


def _majority_train_data():
    # 3 positive out of 10 -> a known, checkable prior of 0.3.
    rows = [_base_row(transaction_amount=float(i)) for i in range(10)]
    labels = np.array([False] * 7 + [True] * 3)
    return _toy_matrix(rows), labels


def _logreg_train_data(n: int = 40):
    rows = []
    labels = []
    for i in range(n):
        label = i % 4 == 0  # 25% positive
        amount = 500.0 if label else 50.0
        rows.append(_base_row(transaction_amount=amount))
        labels.append(label)
    return _toy_matrix(rows), np.array(labels)


# ---------------------------------------------------------------------------
# Model 0: predictions must not depend on feature values
# ---------------------------------------------------------------------------


def test_majority_classifier_predictions_are_independent_of_feature_values():
    X_train, y_train = _majority_train_data()
    model = MajorityClassifier().fit(X_train, y_train, split="train")

    all_zero = _toy_matrix([_base_row() for _ in range(5)])
    large_values = _toy_matrix(
        [_base_row(transaction_amount=v) for v in (1e6, -1e6, 999.0, 0.0001, 42.0)]
    )

    scores_zero = model.predict_proba(all_zero)
    scores_large = model.predict_proba(large_values)

    assert np.allclose(scores_zero, scores_large)
    # And the constant score equals the observed TRAIN prior (3/10 = 0.3).
    assert np.allclose(scores_zero, 0.3)


@pytest.mark.parametrize("split", ["validation", "heldout", "warmup"])
def test_majority_classifier_fit_refused_outside_train(split):
    X_train, y_train = _majority_train_data()
    model = MajorityClassifier()
    with pytest.raises(SplitPolicyError):
        model.fit(X_train, y_train, split=split)
    assert not model.is_fitted


def test_majority_classifier_refit_is_refused():
    X_train, y_train = _majority_train_data()
    model = MajorityClassifier().fit(X_train, y_train, split="train")
    with pytest.raises(SplitPolicyError):
        model.fit(X_train, y_train, split="train")


def test_majority_classifier_predict_before_fit_is_refused():
    matrix = _toy_matrix([_base_row()])
    with pytest.raises(SplitPolicyError):
        MajorityClassifier().predict_proba(matrix)


def test_majority_classifier_rejects_forbidden_columns_at_fit():
    X_train, y_train = _majority_train_data()
    X_train["is_fraud"] = False
    with pytest.raises(LeakageError):
        MajorityClassifier().fit(X_train, y_train, split="train")


# ---------------------------------------------------------------------------
# Model 1: logistic regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["validation", "heldout", "warmup"])
def test_logistic_regression_fit_refused_outside_train(split):
    X_train, y_train = _logreg_train_data()
    model = LogisticRegressionBaseline(random_state=1)
    with pytest.raises(SplitPolicyError):
        model.fit(X_train, y_train, split=split)
    assert not model.is_fitted


def test_logistic_regression_refit_is_refused():
    X_train, y_train = _logreg_train_data()
    model = LogisticRegressionBaseline(random_state=1).fit(X_train, y_train, split="train")
    with pytest.raises(SplitPolicyError):
        model.fit(X_train, y_train, split="train")


def test_logistic_regression_predict_before_fit_is_refused():
    matrix = _toy_matrix([_base_row()])
    with pytest.raises(SplitPolicyError):
        LogisticRegressionBaseline(random_state=1).predict_proba(matrix)


def test_logistic_regression_rejects_forbidden_columns_at_fit():
    X_train, y_train = _logreg_train_data()
    X_train["account_id"] = "acc_1"
    with pytest.raises(LeakageError):
        LogisticRegressionBaseline(random_state=1).fit(X_train, y_train, split="train")


def test_logistic_regression_uses_the_documented_class_weighting_approach():
    model = LogisticRegressionBaseline(random_state=1)
    assert model.hyperparameters["class_weight"] == "balanced"


def test_logistic_regression_is_deterministic_given_a_fixed_seed():
    X_train, y_train = _logreg_train_data()
    a = LogisticRegressionBaseline(random_state=123).fit(X_train, y_train, split="train")
    b = LogisticRegressionBaseline(random_state=123).fit(X_train.copy(), y_train.copy(), split="train")

    coefficients_a = a.coefficients
    coefficients_b = b.coefficients
    assert coefficients_a.keys() == coefficients_b.keys()
    for column in coefficients_a:
        assert coefficients_a[column] == pytest.approx(coefficients_b[column])


def test_logistic_regression_predict_proba_is_deterministic():
    X_train, y_train = _logreg_train_data()
    model = LogisticRegressionBaseline(random_state=1).fit(X_train, y_train, split="train")
    target = _toy_matrix([_base_row(transaction_amount=v) for v in (10.0, 400.0)])

    first = model.predict_proba(target)
    second = model.predict_proba(target)
    assert np.array_equal(first, second)
