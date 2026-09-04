"""Evaluation semantics: cost model, metric primacy, threshold policy.

``EVALUATION.md`` and ``docs/COST_MODEL.md`` define these in prose. This module
makes the parts that can be enforced enforceable, and nothing more — it defines
*what counts as a valid evaluation*, it does not train, score, or fit anything.

Three commitments are worth stating as code rather than as documentation:

**Accuracy is never the primary metric.** At the primary synthetic benchmark
prevalence of 5%, "never fraud" scores 95% accuracy with zero recall.
:data:`NEVER_PRIMARY` records that, and :func:`assert_metric_report_valid` refuses a
report that promotes it.

**Costs are assumptions, not facts.** ``FP_COST = 15`` and ``FN_COST = 450`` are
synthetic benchmark assumptions (:data:`COST_ASSUMPTION`), preserved unchanged from
Checkpoint 0 along with both sensitivity configurations. Absolute cost totals have
no external meaning; only comparisons under identical assumptions do.

**Thresholds come from validation.** :func:`select_threshold` is the only sanctioned
way to choose an operating point and it refuses any split but validation, so
choosing a threshold on held-out data is a raised exception rather than an
oversight.

Not implemented here: PR-AUC, ROC-AUC, and the full cost curve. They are named as
required outputs and computed at Checkpoint 2 in ``ml/evaluation.py``, over model
scores that do not exist yet. Their status is **Not yet measured.**
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml.split_policy import Operation, Split, SplitLike, assert_permitted
from simulator.config import SUPPORTED_PREVALENCES

__all__ = [
    "BENCHMARK_PREVALENCES",
    "COST_ASSUMPTION",
    "COST_CONFIGURATIONS",
    "ConfusionMatrix",
    "CostConfig",
    "DEFAULT_THRESHOLD_NOTE",
    "NEVER_PRIMARY",
    "PREVALENCE_LABEL",
    "PRIMARY_COST",
    "PRIMARY_PREVALENCE",
    "PRIMARY_RANKING_METRIC",
    "REQUIRED_METRICS",
    "SECONDARY_RANKING_METRICS",
    "SENSITIVITY_A",
    "SENSITIVITY_B",
    "THRESHOLD_SELECTION_SPLIT",
    "ThresholdDecision",
    "assert_metric_report_valid",
    "cost_config_from_env",
    "select_threshold",
]


# --------------------------------------------------------------------------
# Cost model (docs/COST_MODEL.md §2)
# --------------------------------------------------------------------------

#: Attach this to every reported cost. It is the caveat that makes the numbers
#: honest rather than merely precise.
COST_ASSUMPTION: str = (
    "Synthetic benchmark assumptions. Not real banking costs, not industry "
    "figures, not derived from any institution's data. Absolute totals carry no "
    "external meaning; only comparisons under identical assumptions do."
)


@dataclass(frozen=True)
class CostConfig:
    """An asymmetric error-cost configuration."""

    name: str
    fp_cost: float
    fn_cost: float

    def __post_init__(self) -> None:
        if self.fp_cost <= 0 or self.fn_cost <= 0:
            raise ValueError(
                f"costs must be positive, got fp={self.fp_cost}, fn={self.fn_cost}"
            )

    @property
    def ratio(self) -> float:
        """``fn_cost / fp_cost`` — how many false positives one miss is worth."""
        return self.fn_cost / self.fp_cost

    def total_cost(self, false_positives: int, false_negatives: int) -> float:
        """``FP × FP_COST + FN × FN_COST``. Lower is better."""
        if false_positives < 0 or false_negatives < 0:
            raise ValueError("error counts cannot be negative")
        return false_positives * self.fp_cost + false_negatives * self.fn_cost


#: The primary configuration. 1:30 — worth accepting ~30 false positives to
#: prevent one miss, which puts the optimal threshold well below 0.5.
PRIMARY_COST = CostConfig("primary", 15.0, 450.0)

#: High-value fraud regime: misses dominate.
SENSITIVITY_A = CostConfig("sensitivity_a", 15.0, 1000.0)

#: Expensive-review regime: scarce investigators raise the price of a false alarm.
SENSITIVITY_B = CostConfig("sensitivity_b", 50.0, 450.0)

#: Every conclusion must survive all three, or the sensitivity must be reported.
COST_CONFIGURATIONS: Tuple[CostConfig, ...] = (
    PRIMARY_COST,
    SENSITIVITY_A,
    SENSITIVITY_B,
)


def cost_config_from_env(env: Optional[Mapping[str, str]] = None) -> CostConfig:
    """Read ``FP_COST`` / ``FN_COST`` from the environment.

    The sanctioned reader, so detection logic never hardcodes a cost
    (``docs/COST_MODEL.md`` §2). Falls back to the primary configuration when the
    variables are absent, and reports itself as ``primary`` when the values match it
    so an experiment record cannot claim a custom configuration it did not use.
    """
    source = env if env is not None else os.environ
    fp = float(source.get("FP_COST", PRIMARY_COST.fp_cost))
    fn = float(source.get("FN_COST", PRIMARY_COST.fn_cost))
    for known in COST_CONFIGURATIONS:
        if (fp, fn) == (known.fp_cost, known.fn_cost):
            return known
    return CostConfig("from_env", fp, fn)


# --------------------------------------------------------------------------
# Metrics (EVALUATION.md)
# --------------------------------------------------------------------------

#: The primary ranking metric for this project.
PRIMARY_RANKING_METRIC: str = "pr_auc"

#: Reported for comparability, never used to rank models — ROC-AUC is optimistic
#: under class imbalance.
SECONDARY_RANKING_METRICS: Tuple[str, ...] = ("roc_auc",)

#: Every model report must carry all of these.
REQUIRED_METRICS: Tuple[str, ...] = (
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "total_error_cost",
)

#: Metrics that may be reported as context and must never be primary.
NEVER_PRIMARY: frozenset = frozenset({"accuracy"})


@dataclass(frozen=True)
class ConfusionMatrix:
    """The full confusion matrix and the rates derived from it.

    Every rate is guarded: an undefined ratio (no predicted positives, no actual
    positives) returns ``0.0`` rather than ``NaN``, so a degenerate model produces a
    reportable row instead of a crash.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @classmethod
    def from_predictions(
        cls, y_true: Sequence[Any], y_pred: Sequence[Any]
    ) -> "ConfusionMatrix":
        truth = np.asarray(y_true).astype(bool)
        pred = np.asarray(y_pred).astype(bool)
        if truth.shape != pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape, got "
                f"{truth.shape} and {pred.shape}"
            )
        return cls(
            true_positives=int(np.sum(truth & pred)),
            false_positives=int(np.sum(~truth & pred)),
            false_negatives=int(np.sum(truth & ~pred)),
            true_negatives=int(np.sum(~truth & ~pred)),
        )

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        return self._ratio(
            self.true_positives, self.true_positives + self.false_positives
        )

    @property
    def recall(self) -> float:
        return self._ratio(
            self.true_positives, self.true_positives + self.false_negatives
        )

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self._ratio(
            self.false_positives, self.false_positives + self.true_negatives
        )

    @property
    def false_negative_rate(self) -> float:
        return self._ratio(
            self.false_negatives, self.false_negatives + self.true_positives
        )

    @property
    def accuracy(self) -> float:
        """Diagnostic only. Listed in :data:`NEVER_PRIMARY` and reported beside the
        majority-classifier baseline so its uselessness here is visible."""
        total = (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )
        return self._ratio(
            self.true_positives + self.true_negatives, total
        )

    def total_error_cost(self, cost: CostConfig = PRIMARY_COST) -> float:
        return cost.total_cost(self.false_positives, self.false_negatives)

    def to_dict(self) -> Dict[str, float]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
        }


def assert_metric_report_valid(
    metrics: Sequence[str], *, primary: str, context: str = "metric report"
) -> None:
    """Fail unless a report carries the required metrics and ranks by a valid one."""
    present = set(metrics)
    missing = [m for m in REQUIRED_METRICS if m not in present]
    if missing:
        raise ValueError(f"{context} is missing required metric(s): {missing}")
    if primary in NEVER_PRIMARY:
        raise ValueError(
            f"{context} promotes {primary!r} to primary. At 5% prevalence a "
            "never-fraud classifier scores 0.95 accuracy with zero recall; "
            f"the primary ranking metric is {PRIMARY_RANKING_METRIC!r} "
            "(EVALUATION.md)."
        )
    if primary != PRIMARY_RANKING_METRIC:
        raise ValueError(
            f"{context} ranks by {primary!r}; this project ranks by "
            f"{PRIMARY_RANKING_METRIC!r}."
        )


# --------------------------------------------------------------------------
# Threshold policy (docs/COST_MODEL.md §3)
# --------------------------------------------------------------------------

#: Thresholds are chosen here and nowhere else.
THRESHOLD_SELECTION_SPLIT: Split = Split.VALIDATION

#: Why 0.5 is not a default worth having.
DEFAULT_THRESHOLD_NOTE: str = (
    "A 0.5 cut encodes a 1:1 error-cost assumption that no configuration in this "
    "project shares. The operating point is chosen by cost minimisation on "
    "validation, never assumed."
)


@dataclass(frozen=True)
class ThresholdDecision:
    """A chosen operating point, inseparable from how it was chosen.

    A threshold reported without its cost configuration and split is not
    interpretable, so all three travel together.
    """

    threshold: float
    split: Split
    cost_config: CostConfig
    total_error_cost: float
    confusion: ConfusionMatrix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "split": self.split.value,
            "cost_config": self.cost_config.name,
            "fp_cost": self.cost_config.fp_cost,
            "fn_cost": self.cost_config.fn_cost,
            "total_error_cost": self.total_error_cost,
            "cost_assumption": COST_ASSUMPTION,
            **self.confusion.to_dict(),
        }


def select_threshold(
    scores: Sequence[float],
    y_true: Sequence[Any],
    *,
    split: SplitLike,
    cost: CostConfig = PRIMARY_COST,
) -> ThresholdDecision:
    """Choose the score cut that minimises total error cost on validation.

    Refuses any split but validation — that guard, not the arithmetic, is why this
    function exists. A transaction is flagged when ``score >= threshold``.

    Candidate thresholds are the distinct observed scores; ties in cost resolve to
    the highest threshold, which is the fewest false positives. "Flag nothing" is
    not a candidate: it is the majority-classifier baseline (Model 0), reported
    separately rather than selected as an operating point.
    """
    assert_permitted(
        Operation.THRESHOLD_SELECTION, split, context="select_threshold"
    )
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y_true).astype(bool)
    if s.shape != y.shape:
        raise ValueError(f"scores and labels differ in shape: {s.shape} vs {y.shape}")
    if s.size == 0:
        raise ValueError("cannot select a threshold from an empty validation set")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must all be finite")

    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    positives = int(y.sum())

    # Flagging the top k rows: cumulative true and false positives at each k.
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    fn = positives - tp
    costs = fp * cost.fp_cost + fn * cost.fn_cost

    # Only cut between distinct scores; flagging half of a tied group is not a
    # threshold anyone could apply.
    boundary = np.empty(s_sorted.size, dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]
    boundary[-1] = True
    candidate_costs = np.where(boundary, costs, np.inf)
    k = int(np.argmin(candidate_costs))  # first minimum == highest threshold

    threshold = float(s_sorted[k])
    confusion = ConfusionMatrix.from_predictions(y, s >= threshold)
    return ThresholdDecision(
        threshold=threshold,
        split=Split(THRESHOLD_SELECTION_SPLIT),
        cost_config=cost,
        total_error_cost=confusion.total_error_cost(cost),
        confusion=confusion,
    )


# --------------------------------------------------------------------------
# Benchmark configurations (docs/ML_READINESS.md §4)
# --------------------------------------------------------------------------

#: Preserved unchanged from Checkpoint 1: 1%, 5%, 10%, 25%.
BENCHMARK_PREVALENCES: Tuple[float, ...] = tuple(SUPPORTED_PREVALENCES)

#: The primary synthetic benchmark configuration — a benchmark choice, explicitly
#: not a real-world or industry fraud rate.
PRIMARY_PREVALENCE: float = 0.05

PREVALENCE_LABEL: str = "primary synthetic benchmark configuration"
