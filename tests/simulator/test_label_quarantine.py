"""9. Quarantine / label safety — the one-way barrier holds.

The dataset carries its own answer key: ``is_fraud`` and ``attack_id`` on every row,
plus entity flags (``Merchant.is_fake``, ``Customer.is_synthetic_identity``, the
``is_shared_legitimate`` markers) that exist so an evaluator can score a detector. Those
fields make evaluation possible and would make modelling meaningless.

Two things therefore have to be true, and they are different claims.

*The feature-safe view is genuinely stripped.* ``feature_safe_row`` must expose every
non-label column and no label column, and the labels must be **absent** rather than
blanked — a downstream typo should raise ``KeyError``, not quietly read ``None``.

*The tripwire actually fires.* A test that only checks the safe path would pass just as
well against an ``assert_feature_safe`` that never raised. So each quarantined name is
fed in deliberately, bare and in the ``.``-qualified and ``_``-prefixed forms a real join
produces, and must raise :class:`LeakageError`.

The audit's own negative control is checked too. ``_probe_attack_specific_fields``
measures the quarantined flags on purpose — ``merchant.is_fake`` predicts
``fake_merchant_ring`` at AUC ≈ 0.97, ``customer.is_synthetic_identity`` predicts
``synthetic_identity_cluster`` at AUC ≈ 0.99 — and records them as ``quarantined``
rather than as failures, precisely because they are the labels. That measurement is what
justifies the quarantine instead of leaving it an unexamined assumption, so this file
asserts it is still being taken, and still not counted as an oracle.
"""

from __future__ import annotations

import pytest

from simulator.leakage import (
    ENTITY_LABEL_FIELDS,
    FEATURE_SAFE_FIELDS,
    LABEL_FIELDS,
    LeakageError,
    assert_feature_safe,
    assert_no_label_fields,
    feature_safe_row,
)
from simulator.transactions import TRANSACTION_FIELDS

#: ``docs/FEATURE_LEAKAGE_POLICY.md`` §4 — these guards must never be skipped.
pytestmark = pytest.mark.leakage

#: Every quarantined name, in the forms a careless join actually produces.
LEAKY_COLUMN_FORMS = (
    ["is_fraud"],
    ["attack_id"],
    ["merchant.is_fake"],
    ["merchant_is_fake"],
    ["customer.is_synthetic_identity"],
    ["device_is_shared_legitimate"],
    ["amount", "channel", "is_fraud"],  # hidden among legitimate columns
    list(TRANSACTION_FIELDS),  # the raw schema, unfiltered
)


def test_quarantined_field_names_are_frozen():
    """A freeze guard: the barrier is only as wide as this list."""
    assert LABEL_FIELDS == frozenset({"is_fraud", "attack_id"})
    assert set(ENTITY_LABEL_FIELDS) == {"customer", "merchant", "device", "ip"}
    assert ENTITY_LABEL_FIELDS["merchant"] == frozenset({"is_fake"})
    assert ENTITY_LABEL_FIELDS["customer"] == frozenset({"is_synthetic_identity"})


def test_feature_safe_fields_are_the_schema_minus_the_labels():
    """Derived by subtraction, so a new column cannot be silently omitted."""
    assert set(FEATURE_SAFE_FIELDS) == set(TRANSACTION_FIELDS) - LABEL_FIELDS
    assert not LABEL_FIELDS & set(FEATURE_SAFE_FIELDS)
    assert len(FEATURE_SAFE_FIELDS) == len(TRANSACTION_FIELDS) - len(LABEL_FIELDS)


def test_feature_safe_fields_pass_the_barrier():
    """The positive case: the sanctioned column list must not trip the tripwire."""
    assert_feature_safe(FEATURE_SAFE_FIELDS, context="feature-safe field list")


@pytest.mark.parametrize("columns", LEAKY_COLUMN_FORMS)
def test_quarantined_columns_are_rejected(columns):
    """The negative control: without this, a no-op barrier would pass every test."""
    with pytest.raises(LeakageError):
        assert_feature_safe(columns, context="test")


def test_feature_safe_row_omits_the_labels_entirely(dataset):
    """Absent, not blanked — a typo must raise ``KeyError``, not read ``None``."""
    fraud_row = next(t for t in dataset.transactions if t.is_fraud)
    normal_row = next(t for t in dataset.transactions if not t.is_fraud)

    for txn in (fraud_row, normal_row):
        safe = feature_safe_row(txn)

        assert set(safe) == set(FEATURE_SAFE_FIELDS)
        assert not LABEL_FIELDS & set(safe)
        for label in LABEL_FIELDS:
            with pytest.raises(KeyError):
                safe[label]


def test_raw_rows_are_rejected_but_stripped_rows_pass(dataset):
    """``assert_no_label_fields`` must distinguish the row from its safe view."""
    txn = dataset.transactions[0]

    with pytest.raises(LeakageError):
        assert_no_label_fields(txn, context="raw transaction")

    assert_no_label_fields(feature_safe_row(txn), context="feature-safe row")


def test_entity_objects_are_rejected_by_the_barrier(dataset):
    """Entity flags are the realistic leak — a join, not a deliberate read."""
    eco = dataset.ecosystem

    for entity in (eco.merchants[0], eco.customers[0], eco.devices[0], eco.ips[0]):
        with pytest.raises(LeakageError):
            assert_no_label_fields(entity, context=type(entity).__name__)


def test_audit_still_measures_the_quarantined_fields_without_failing(audit_for):
    """The audit's negative control: quantified, recorded, never counted as an oracle."""
    report = audit_for()

    quarantined = [r for r in report.results if r.verdict == "quarantined"]

    assert quarantined, "the audit no longer measures the quarantined label fields"
    assert all(r.kind == "quarantined_field" for r in quarantined)
    assert not any(r.is_oracle for r in quarantined)
    # The fields are the answer key, so at least one must in fact be near-perfect —
    # otherwise the justification for quarantining them has quietly evaporated.
    assert max(r.auc for r in quarantined if r.auc is not None) >= 0.90
