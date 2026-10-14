"""Label quarantine: the one-way barrier between ground truth and features.

The dataset deliberately carries its own answer key. ``Transaction.is_fraud`` and
``Transaction.attack_id`` sit on every row, and several entities carry flags that
exist purely so the evaluator can score a detector: ``Customer``
``is_synthetic_identity``, ``Merchant.is_fake``, and the ``is_shared_legitimate``
markers that distinguish benign device/IP sharing from a ring.

Those fields make evaluation possible and would make modelling meaningless. The
shortcut audit measures exactly how meaningless: ``customer.is_synthetic_identity``
predicts ``synthetic_identity_cluster`` at AUC 0.9906 and ``merchant.is_fake``
predicts ``fake_merchant_ring`` at AUC 0.9655. Both are reported as ``quarantined``
rather than as failures, because they are *supposed* to be perfect — they are the
labels. The failure mode is not that they separate fraud; it is a feature pipeline
reading them by accident.

This module is that accident's tripwire. It names the quarantined fields in one
place and offers a row view that physically cannot expose them, so feature code
never has to remember the rule.

**Direction of the barrier.** Evaluation-side code (``validation.py``,
``shortcut_audit.py``, metric computation) may read labels freely — that is its
job. Feature, model, and scoring code may not, and must never import from those
modules. The dependency is one-way by design, and
:func:`assert_no_label_fields` exists so a test can assert it rather than trusting
a convention (``docs/FEATURE_LEAKAGE_POLICY.md`` §4).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Set

from simulator.transactions import TRANSACTION_FIELDS

__all__ = [
    "ENTITY_LABEL_FIELDS",
    "FEATURE_SAFE_FIELDS",
    "LABEL_FIELDS",
    "LeakageError",
    "assert_feature_safe",
    "assert_no_label_fields",
    "feature_safe_row",
    "feature_safe_rows",
]


class LeakageError(AssertionError):
    """Raised when ground truth reaches something that must not see it.

    Deliberately an ``AssertionError`` subclass: this is a broken invariant, not a
    recoverable condition. Nothing should ever catch it and continue.
    """


#: Transaction-level ground truth. Present on every row, readable only by
#: evaluation code.
LABEL_FIELDS: frozenset = frozenset({"is_fraud", "attack_id"})

#: Entity-level ground truth, keyed by the entity type that carries it. These are
#: not on the transaction row, but a naive join would pull them in — which is the
#: realistic way this leak happens.
ENTITY_LABEL_FIELDS: Mapping[str, frozenset] = {
    "customer": frozenset({"is_synthetic_identity"}),
    "merchant": frozenset({"is_fake"}),
    "device": frozenset({"is_shared_legitimate"}),
    "ip": frozenset({"is_shared_legitimate"}),
}

#: Every transaction column a feature pipeline may legitimately read. Derived by
#: subtraction from :data:`simulator.transactions.TRANSACTION_FIELDS` rather than
#: retyped, so adding a column to the schema cannot silently omit it here — and
#: adding a *label* column only requires updating :data:`LABEL_FIELDS`.
FEATURE_SAFE_FIELDS: tuple = tuple(
    name for name in TRANSACTION_FIELDS if name not in LABEL_FIELDS
)


def _all_label_names() -> Set[str]:
    names: Set[str] = set(LABEL_FIELDS)
    for fields in ENTITY_LABEL_FIELDS.values():
        names |= set(fields)
    return names


def feature_safe_row(txn: Any) -> Dict[str, Any]:
    """Project one transaction down to its feature-safe columns.

    Returns a plain dict containing only :data:`FEATURE_SAFE_FIELDS`. The labels are
    not blanked or set to ``None`` — they are *absent*, so a downstream typo raises
    ``KeyError`` instead of quietly training on the answer.
    """
    return {name: getattr(txn, name) for name in FEATURE_SAFE_FIELDS}


def feature_safe_rows(transactions: Iterable[Any]) -> list:
    """Vectorised :func:`feature_safe_row`."""
    return [feature_safe_row(t) for t in transactions]


def assert_feature_safe(columns: Sequence[str], *, context: str = "feature matrix") -> None:
    """Fail if ``columns`` contains any quarantined field name.

    Call this at the boundary where a feature matrix is handed to a model. Checks
    entity-level flags as well as transaction labels, since the common leak is a
    join that drags ``is_fake`` along rather than anyone reading ``is_fraud``
    directly.

    Suffix-matching on ``.``-qualified names means prefixed columns are caught too:
    a join producing ``merchant.is_fake`` or ``merchant_is_fake`` fails just as a
    bare ``is_fake`` would.
    """
    banned = _all_label_names()
    found = []
    for col in columns:
        tail = col.split(".")[-1]
        if col in banned or tail in banned:
            found.append(col)
            continue
        if any(col.endswith(f"_{b}") or col == b for b in banned):
            found.append(col)
    if found:
        raise LeakageError(
            f"{context} exposes quarantined ground-truth field(s): {sorted(set(found))}. "
            "These are the answer key (see docs/FEATURE_LEAKAGE_POLICY.md §4); "
            "read them only from evaluation code."
        )


def assert_no_label_fields(obj: Any, *, context: str = "object") -> None:
    """Fail if ``obj`` exposes quarantined attributes.

    For asserting that a row view handed to feature code really is stripped — the
    complement of :func:`assert_feature_safe`, which checks column *names*.
    """
    if isinstance(obj, Mapping):
        assert_feature_safe(list(obj.keys()), context=context)
        return
    present = [name for name in _all_label_names() if hasattr(obj, name)]
    if present:
        raise LeakageError(
            f"{context} exposes quarantined ground-truth attribute(s): {sorted(present)}"
        )
