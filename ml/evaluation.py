"""Metric computation over model scores (EVALUATION.md, ml/evaluation_contract.py).

``ml/evaluation_contract.py`` defines what a valid evaluation *is* -- the
primary metric, the required metric set, the cost model, and the
validation-only threshold rule -- but deliberately computes nothing (its own
docstring: "Not implemented here: PR-AUC, ROC-AUC, and the full cost curve.
... computed at Checkpoint 2 in ml/evaluation.py"). This module is that
computation. It never redefines the cost model, the primary metric, or the
threshold rule -- all of that is imported, unchanged, from
``ml.evaluation_contract``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evaluation_contract import (
    PRIMARY_COST,
    PRIMARY_RANKING_METRIC,
    CostConfig,
    ThresholdDecision,
    assert_metric_report_valid,
    select_threshold,
)
from ml.split_policy import Split

__all__ = ["ModelReport", "evaluate_on_validation"]


class ModelReport:
    """The full, contract-valid metrics report for one model on validation."""

    def __init__(
        self,
        *,
        model_type: str,
        pr_auc: float,
        roc_auc: float,
        threshold_decision: ThresholdDecision,
    ) -> None:
        self.model_type = model_type
        self.pr_auc = pr_auc
        self.roc_auc = roc_auc
        self.threshold_decision = threshold_decision

    def to_dict(self) -> dict[str, Any]:
        confusion = self.threshold_decision.confusion
        metrics: dict[str, Any] = {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "precision": confusion.precision,
            "recall": confusion.recall,
            "f1": confusion.f1,
            "confusion_matrix": confusion.to_dict(),
            "total_error_cost": self.threshold_decision.total_error_cost,
        }
        # Re-asserted here, not just trusted: a report that ever tried to
        # promote roc_auc (or anything but pr_auc) to primary is refused.
        assert_metric_report_valid(
            list(metrics), primary=PRIMARY_RANKING_METRIC, context=f"{self.model_type} report"
        )
        return {
            "model_type": self.model_type,
            "metrics": metrics,
            "threshold": self.threshold_decision.to_dict(),
        }


def evaluate_on_validation(
    y_true: Sequence[Any],
    scores: Sequence[float],
    *,
    model_type: str,
    cost: CostConfig = PRIMARY_COST,
) -> ModelReport:
    """Score one model's validation predictions.

    Computes PR-AUC (primary) and ROC-AUC (secondary diagnostic only -- never
    promotable to primary, enforced in :meth:`ModelReport.to_dict`), then
    selects the cost-minimising operating threshold via
    :func:`ml.evaluation_contract.select_threshold`.

    Deliberately has no ``split`` parameter: this function *is* "evaluate on
    validation", so there is no argument through which a caller could ever
    point it at train or held-out. ``select_threshold`` itself would refuse
    any other split regardless (``ml.split_policy.Operation.THRESHOLD_SELECTION``
    permits only ``VALIDATION``), which is a second, independent guard.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"y_true and scores differ in shape: {y.shape} vs {s.shape}")

    pr_auc = float(average_precision_score(y, s))
    roc_auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    decision = select_threshold(s, y, split=Split.VALIDATION, cost=cost)
    return ModelReport(
        model_type=model_type, pr_auc=pr_auc, roc_auc=roc_auc, threshold_decision=decision
    )
