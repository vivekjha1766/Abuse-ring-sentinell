"""Evaluation semantics: costs frozen, accuracy demoted, thresholds from validation.

``EVALUATION.md`` and ``docs/COST_MODEL.md`` make three commitments that can be
tested without a model, and all three are things that go wrong quietly rather than
loudly:

*A cost constant drifts* and every earlier comparison silently stops being
comparable, so the three configurations are asserted literally.

*Accuracy is promoted* because it is the highest number on the page. The refusal is
tested against the exact degenerate classifier that makes it tempting: never-fraud
at 5% prevalence, which scores 0.95 accuracy with zero recall.

*A threshold is picked on the test set*, usually while "just checking". Threshold
selection is therefore tested for its refusals as much as for its arithmetic, and
its arithmetic is checked against an independent brute-force search rather than
against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.evaluation_contract import (
    BENCHMARK_PREVALENCES,
    COST_CONFIGURATIONS,
    NEVER_PRIMARY,
    PRIMARY_COST,
    PRIMARY_PREVALENCE,
    PRIMARY_RANKING_METRIC,
    REQUIRED_METRICS,
    SENSITIVITY_A,
    SENSITIVITY_B,
    THRESHOLD_SELECTION_SPLIT,
    ConfusionMatrix,
    CostConfig,
    assert_metric_report_valid,
    cost_config_from_env,
    select_threshold,
)
from ml.split_policy import Split, SplitPolicyError

pytestmark = pytest.mark.leakage


def test_cost_configurations_are_frozen():
    """``docs/COST_MODEL.md`` §2, unchanged since Checkpoint 0."""
    assert (PRIMARY_COST.fp_cost, PRIMARY_COST.fn_cost) == (15.0, 450.0)
    assert (SENSITIVITY_A.fp_cost, SENSITIVITY_A.fn_cost) == (15.0, 1000.0)
    assert (SENSITIVITY_B.fp_cost, SENSITIVITY_B.fn_cost) == (50.0, 450.0)
    assert PRIMARY_COST.ratio == 30.0
    assert COST_CONFIGURATIONS == (PRIMARY_COST, SENSITIVITY_A, SENSITIVITY_B)


def test_total_error_cost_arithmetic():
    """``FP × FP_COST + FN × FN_COST``, and negative counts are a bug not a discount."""
    assert PRIMARY_COST.total_cost(10, 2) == 10 * 15.0 + 2 * 450.0
    assert PRIMARY_COST.total_cost(0, 0) == 0.0
    with pytest.raises(ValueError):
        PRIMARY_COST.total_cost(-1, 0)
    with pytest.raises(ValueError):
        CostConfig("bad", 0.0, 450.0)


def test_costs_are_read_from_the_environment():
    """The sanctioned reader, so no detection path hardcodes a cost."""
    assert cost_config_from_env({}) is PRIMARY_COST
    assert cost_config_from_env({"FP_COST": "15", "FN_COST": "1000"}) is SENSITIVITY_A
    custom = cost_config_from_env({"FP_COST": "7", "FN_COST": "99"})
    assert (custom.name, custom.fp_cost, custom.fn_cost) == ("from_env", 7.0, 99.0)


def test_confusion_matrix_and_derived_rates():
    """One worked example, checked by hand."""
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0]
    cm = ConfusionMatrix.from_predictions(y_true, y_pred)

    assert (cm.true_positives, cm.false_positives) == (2, 1)
    assert (cm.false_negatives, cm.true_negatives) == (1, 4)
    assert cm.precision == pytest.approx(2 / 3)
    assert cm.recall == pytest.approx(2 / 3)
    assert cm.f1 == pytest.approx(2 / 3)
    assert cm.false_positive_rate == pytest.approx(0.2)
    assert cm.false_negative_rate == pytest.approx(1 / 3)
    assert cm.total_error_cost(PRIMARY_COST) == 15.0 + 450.0

    with pytest.raises(ValueError):
        ConfusionMatrix.from_predictions([1, 0], [1, 0, 1])


def test_rates_stay_finite_for_a_degenerate_classifier():
    """A flag-nothing model must produce a reportable row, not a ``NaN``."""
    cm = ConfusionMatrix.from_predictions([1, 0, 0, 0], [0, 0, 0, 0])
    for value in (cm.precision, cm.recall, cm.f1, cm.false_positive_rate):
        assert math.isfinite(value)
    assert cm.precision == 0.0
    assert cm.recall == 0.0
    assert cm.f1 == 0.0


def test_accuracy_is_never_the_primary_metric():
    """The concrete reason: 95% accuracy, zero recall, zero value."""
    y_true = [1] * 5 + [0] * 95
    never_fraud = [0] * 100
    cm = ConfusionMatrix.from_predictions(y_true, never_fraud)

    assert cm.accuracy == pytest.approx(0.95)
    assert cm.recall == 0.0
    assert "accuracy" in NEVER_PRIMARY

    with pytest.raises(ValueError):
        assert_metric_report_valid([*REQUIRED_METRICS, "accuracy"], primary="accuracy")
    assert_metric_report_valid(REQUIRED_METRICS, primary=PRIMARY_RANKING_METRIC)


def test_a_report_missing_a_required_metric_is_refused():
    """Precision without recall, or either without the cost, is not a result."""
    partial = [m for m in REQUIRED_METRICS if m != "recall"]
    with pytest.raises(ValueError):
        assert_metric_report_valid(partial, primary=PRIMARY_RANKING_METRIC)
    with pytest.raises(ValueError):
        assert_metric_report_valid(REQUIRED_METRICS, primary="roc_auc")


@pytest.mark.parametrize("split", ["train", "heldout", "warmup"])
def test_threshold_cannot_be_selected_off_validation(split):
    """The guard that is the whole reason ``select_threshold`` exists."""
    rng = np.random.default_rng(42)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.05).astype(int)

    with pytest.raises(SplitPolicyError):
        select_threshold(scores, labels, split=split)


def _brute_force_best_cost(scores, labels, cost=PRIMARY_COST) -> float:
    """The cheapest applicable threshold, found by trying all of them.

    Independent of the sweep under test: it applies each candidate as a caller
    would (``score >= threshold``) and rebuilds the confusion matrix from scratch,
    so it cannot inherit an off-by-one from the implementation it checks.
    """
    scores = np.asarray(scores, dtype=float)
    best = math.inf
    for threshold in np.unique(scores):
        cm = ConfusionMatrix.from_predictions(labels, scores >= threshold)
        best = min(best, cost.total_cost(cm.false_positives, cm.false_negatives))
    return best


def test_threshold_minimises_cost_and_records_how_it_was_chosen():
    """Checked against an independent brute-force sweep, not against itself."""
    rng = np.random.default_rng(7)
    labels = (rng.random(400) < 0.05).astype(int)
    # Scores correlated with the labels, so the optimum is interior rather than at
    # an edge — an edge optimum would pass even a broken sweep.
    scores = np.clip(0.25 * labels + rng.normal(0.4, 0.15, 400), 0.0, 1.0)

    decision = select_threshold(scores, labels, split="validation", cost=PRIMARY_COST)

    assert decision.total_error_cost == pytest.approx(
        _brute_force_best_cost(scores, labels)
    )
    assert decision.split is Split.VALIDATION
    assert decision.cost_config is PRIMARY_COST
    assert decision.confusion.total_error_cost(PRIMARY_COST) == decision.total_error_cost

    recorded = decision.to_dict()
    assert recorded["split"] == "validation"
    assert recorded["fp_cost"] == 15.0
    assert "Synthetic benchmark assumptions" in recorded["cost_assumption"]


def test_tied_scores_do_not_produce_a_phantom_optimum():
    """Heavy ties are where a cumulative sweep goes wrong.

    A sweep that may stop part-way through a tied group sees a cost that no
    applicable threshold can deliver: here, "flag only the one fraud among the 41
    rows scored 0.9" costs 450 on paper, but the threshold 0.9 actually flags all 41
    and costs 1050. The genuinely cheapest applicable cut is 0.5.

    Both properties are asserted, because tie-consistency alone passes even when the
    sweep is wrong — the reported threshold is re-applied afterwards, so the flag set
    is always consistent while the *choice* is not.
    """
    scores = [0.9] * 41 + [0.5] + [0.1] * 8
    labels = [1] + [0] * 40 + [1] + [0] * 8

    decision = select_threshold(scores, labels, split="validation", cost=PRIMARY_COST)

    assert decision.total_error_cost == pytest.approx(
        _brute_force_best_cost(scores, labels)
    )
    assert decision.threshold == 0.5

    flagged = [s >= decision.threshold for s in scores]
    for score in set(scores):
        group = {f for s, f in zip(scores, flagged) if s == score}
        assert len(group) == 1, f"score {score} was split across the threshold"


def test_threshold_selection_refuses_empty_or_non_finite_scores():
    with pytest.raises(ValueError):
        select_threshold([], [], split="validation")
    with pytest.raises(ValueError):
        select_threshold([0.5, float("nan")], [1, 0], split="validation")


def test_benchmark_prevalences_are_preserved():
    """Checkpoint 1's four configurations, with 5% named as primary."""
    assert BENCHMARK_PREVALENCES == (0.01, 0.05, 0.10, 0.25)
    assert PRIMARY_PREVALENCE == 0.05
    assert THRESHOLD_SELECTION_SPLIT is Split.VALIDATION
