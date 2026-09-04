"""6. Point-in-time / causality invariants — nothing is referenced before it exists.

A transaction that names an entity created *after* the transaction happened is not a
cosmetic oddity: the entity's age is a feature, and a negative age is information that
could not have been observed at scoring time. A model would happily exploit it, and the
resulting metrics would be fiction. ``docs/FEATURE_LEAKAGE_POLICY.md`` §1 therefore
treats this as an error rather than a warning, and this file guards it directly — an
earlier iteration of the generator produced accounts with ages as negative as −108 days,
which is precisely the regression being pinned.

The reference list is imported from :mod:`simulator.validation` rather than restated, so
that adding a seventh referenced entity type there cannot leave this test silently
checking six.

Two things are deliberately *not* asserted:

* **Device/IP adoption timing** — that an account only transacts on a device it had
  already adopted. Attack generators emit deliberate cover traffic
  (``is_fraud=False``, ``attack_id=None``) on freshly minted devices, so the rule is
  not uniform across normal rows and asserting it would manufacture a failure rather
  than find one.
* **Ecosystem link-table timing** — the link tables record the relationships as of the
  end of generation, not a history, so there is no per-link timestamp to compare.
"""

from __future__ import annotations

import pytest

from simulator.validation import _CAUSAL_REFERENCES

#: ``docs/FEATURE_LEAKAGE_POLICY.md`` §1 — point-in-time guards, never skipped.
pytestmark = pytest.mark.leakage


@pytest.mark.parametrize("field_name,lookup_attr,label", _CAUSAL_REFERENCES)
def test_no_transaction_references_an_entity_created_later(
    dataset, field_name, lookup_attr, label
):
    lookup = getattr(dataset.ecosystem, lookup_attr)

    violations = []
    worst = 0.0
    for txn in dataset.transactions:
        entity_id = getattr(txn, field_name)
        if entity_id is None:
            continue
        entity = lookup.get(entity_id)
        if entity is None:
            continue  # a dangling id is entity-integrity's finding, not causality's
        age_days = (txn.timestamp - entity.created_at).total_seconds() / 86400.0
        if age_days < 0.0:
            worst = min(worst, age_days)
            if len(violations) < 5:
                violations.append((txn.transaction_id, entity_id, round(age_days, 2)))

    assert not violations, (
        f"transactions reference a {label} created after them (most negative age "
        f"{worst:.1f} days): {violations}"
    )


@pytest.mark.parametrize("field_name,lookup_attr,label", _CAUSAL_REFERENCES)
def test_no_entity_is_created_after_the_observation_window_closes(
    dataset, field_name, lookup_attr, label
):
    """History invented beyond the dataset's own end would be unobservable."""
    horizon = dataset.config.end
    lookup = getattr(dataset.ecosystem, lookup_attr)

    late = [
        entity_id
        for entity_id, entity in lookup.items()
        if entity.created_at > horizon
    ]

    assert not late, (
        f"{len(late)} {label} entities created after {horizon.isoformat()}: {late[:5]}"
    )


def test_accounts_are_never_older_than_their_customer(dataset):
    """An account cannot predate the customer who owns it."""
    eco = dataset.ecosystem
    customers = eco.customer_by_id

    impossible = [
        (a.account_id, a.created_at.isoformat(), customers[a.customer_id].created_at.isoformat())
        for a in eco.accounts
        if a.customer_id in customers and a.created_at < customers[a.customer_id].created_at
    ]

    assert not impossible, f"accounts predating their customer: {impossible[:5]}"


def test_no_future_entity_is_available_to_a_transaction(dataset):
    """The strong form of the same rule, stated as a set inclusion.

    For every row, the entities it names must all be drawn from the population that
    already existed at its timestamp. Stating it this way catches a generator that
    samples from the *final* entity pool instead of the point-in-time one, even if the
    per-field ages happened to come out non-negative.
    """
    eco = dataset.ecosystem
    lookups = {attr: getattr(eco, attr) for _, attr, _ in _CAUSAL_REFERENCES}

    offenders = []
    for txn in dataset.transactions:
        for field_name, attr, label in _CAUSAL_REFERENCES:
            entity_id = getattr(txn, field_name)
            if entity_id is None:
                continue
            entity = lookups[attr].get(entity_id)
            if entity is not None and entity.created_at > txn.timestamp:
                offenders.append((txn.transaction_id, label, entity_id))
                break
        if len(offenders) >= 5:
            break

    assert not offenders, f"rows drawing on future entities: {offenders}"


def test_attack_records_never_start_before_the_dataset_begins(dataset):
    """Attack windows must sit inside the simulated history, not before it."""
    cfg = dataset.config

    outside = [
        record.attack_id
        for record in dataset.ground_truth.records
        if record.start_time < cfg.warmup_start or record.end_time > cfg.end
    ]

    assert not outside, f"attack windows outside the simulated history: {outside[:5]}"
