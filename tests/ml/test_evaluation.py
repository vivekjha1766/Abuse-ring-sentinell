"""Checkpoint 2E: correctness tests for ml/evaluation.py.

Covers: PR-AUC/ROC-AUC are computed correctly against small, hand-checked
score/label arrays; the assembled report always uses pr_auc as primary and
never roc_auc or accuracy; shape mismatches are refused; the primary cost
configuration is used by default.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evaluation import evaluate_on_validation
from ml.evaluation_contract import PRIMARY_COST, assert_metric_report_valid

pytestmark = pytest.mark.ml


def _separable_scores():
    # 10 rows, 3 positive, scores perfectly ranked by label.
    y_true = [False] * 7 + [True] * 3
    scores = [0.05, 0.1, 0.12, 0.2, 0.22, 0.3, 0.35, 0.8, 0.85, 0.9]
    return y_true, scores


def test_pr_auc_and_roc_auc_match_sklearn_directly():
    y_true, scores = _separable_scores()
    report = evaluate_on_validation(y_true, scores, model_type="test_model")

    assert report.pr_auc == pytest.approx(average_precision_score(y_true, scores))
    assert report.roc_auc == pytest.approx(roc_auc_score(y_true, scores))


def test_report_contains_all_required_metrics_with_pr_auc_primary():
    y_true, scores = _separable_scores()
    report = evaluate_on_validation(y_true, scores, model_type="test_model")
    payload = report.to_dict()  # must not raise

    metrics = payload["metrics"]
    for required in ("pr_auc", "precision", "recall", "f1", "confusion_matrix", "total_error_cost"):
        assert required in metrics
    assert "roc_auc" in metrics  # present, but only as a secondary diagnostic


def test_contract_still_refuses_promoting_roc_auc_or_accuracy_as_primary():
    y_true, scores = _separable_scores()
    report = evaluate_on_validation(y_true, scores, model_type="test_model")
    metric_names = list(report.to_dict()["metrics"])

    with pytest.raises(ValueError):
        assert_metric_report_valid(metric_names, primary="roc_auc")
    with pytest.raises(ValueError):
        assert_metric_report_valid(metric_names, primary="accuracy")


def test_shape_mismatch_between_labels_and_scores_is_refused():
    with pytest.raises(ValueError):
        evaluate_on_validation([True, False, True], [0.1, 0.9], model_type="test_model")


def test_default_cost_configuration_is_the_primary_one():
    y_true, scores = _separable_scores()
    report = evaluate_on_validation(y_true, scores, model_type="test_model")
    assert report.threshold_decision.cost_config is PRIMARY_COST


def test_threshold_decision_is_selected_on_validation():
    y_true, scores = _separable_scores()
    report = evaluate_on_validation(y_true, scores, model_type="test_model")
    assert report.threshold_decision.split.value == "validation"


def test_roc_auc_is_nan_when_only_one_class_present():
    y_true = [False] * 5
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    report = evaluate_on_validation(y_true, scores, model_type="degenerate")
    assert np.isnan(report.roc_auc)
