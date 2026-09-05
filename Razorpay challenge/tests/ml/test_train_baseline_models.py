"""Checkpoint 2E follow-up: the experiment record's hyperparameters field.

``scripts/train_baseline_models.py`` persists each model's hyperparameters by
reading its own live configuration -- ``ml.models.LogisticRegressionBaseline``'s
public ``.hyperparameters`` property, and, since ``MajorityClassifier`` exposes
no such property, its wrapped ``sklearn.dummy.DummyClassifier``'s own
``get_params()``. These tests exercise that exact construction against real
model instances, so the persisted record's "hyperparameters" field is verified
truthful and non-hardcoded without needing a real dataset on disk.

These are mechanism-shape tests only, not a demonstration of split policy
(already covered by tests/leakage/test_models.py) -- labels here are reused
as both fit target and "evaluation" target purely to exercise
evaluate_on_validation's output shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation import evaluate_on_validation
from ml.feature_registry import FEATURE_COLUMNS
from ml.models import LogisticRegressionBaseline, MajorityClassifier

pytestmark = pytest.mark.ml


def _base_row(**overrides: float) -> dict[str, float]:
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row.update(overrides)
    return row


def _toy_matrix(rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame[list(FEATURE_COLUMNS)]


def _toy_data(n: int = 20):
    rows = []
    labels = []
    for i in range(n):
        label = i % 5 == 0
        rows.append(_base_row(transaction_amount=200.0 if label else 20.0))
        labels.append(label)
    return _toy_matrix(rows), np.array(labels)


def test_majority_classifier_record_entry_contains_truthful_hyperparameters():
    X, y = _toy_data()
    model = MajorityClassifier().fit(X, y, split="train")
    report = evaluate_on_validation(y, model.predict_proba(X), model_type="majority")

    # The exact construction scripts/train_baseline_models.py uses.
    record_entry = {**report.to_dict(), "hyperparameters": model._model.get_params()}

    assert "hyperparameters" in record_entry
    assert record_entry["hyperparameters"]["strategy"] == "prior"


def test_logistic_regression_record_entry_contains_truthful_hyperparameters():
    X, y = _toy_data()
    model = LogisticRegressionBaseline(random_state=7).fit(X, y, split="train")
    report = evaluate_on_validation(
        y, model.predict_proba(X), model_type="logistic_regression"
    )

    # The exact construction scripts/train_baseline_models.py uses.
    record_entry = {**report.to_dict(), "hyperparameters": model.hyperparameters}

    assert "hyperparameters" in record_entry
    assert record_entry["hyperparameters"]["class_weight"] == "balanced"
    assert record_entry["hyperparameters"]["random_state"] == 7
