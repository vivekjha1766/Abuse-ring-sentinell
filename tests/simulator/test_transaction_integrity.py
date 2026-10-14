"""3. Transaction integrity — identifiers, value ranges, and timestamps.

The row schema is the contract every later checkpoint builds on, so this file pins the
parts of it that silent generator drift would break first: identifier uniqueness,
amounts that are positive and expressed in whole cents, categorical fields drawn from
the declared vocabularies, coordinates on the globe, and timestamps that are timezone
aware and inside the dataset's own observation window.

Rounding is asserted because it is a deliberate anti-artifact measure rather than
cosmetic: uniformly round amounts on fraud rows would be a giveaway, so attack amounts
pass through the same continuous draw and two-decimal rounding as legitimate traffic
(:func:`simulator.attacks.base.jittered_amount`).
"""

from __future__ import annotations

import dataclasses
import math

from simulator.transactions import (
    CHANNELS,
    STATUSES,
    TRANSACTION_FIELDS,
    TRANSACTION_TYPES,
    Transaction,
)


def test_transaction_ids_are_unique(dataset):
    ids = [t.transaction_id for t in dataset.transactions]

    assert len(ids) == len(set(ids)), "duplicate transaction_id values"
    assert all(isinstance(i, str) and i for i in ids)


def test_transaction_fields_cover_the_dataclass(dataset):
    """``TRANSACTION_FIELDS`` drives the manifest hash and the quarantine list.

    It is an *export* order, not the declaration order — ground-truth columns are
    deliberately last so they can be dropped as a block — so only the set has to
    match, plus that block placement.
    """
    declared = {f.name for f in dataclasses.fields(Transaction)}

    assert set(TRANSACTION_FIELDS) == declared
    assert len(TRANSACTION_FIELDS) == len(declared), "duplicate column in export order"
    assert TRANSACTION_FIELDS[-2:] == ("is_fraud", "attack_id")


def test_amounts_are_positive_finite_and_rounded_to_cents(dataset):
    bad_value = [
        t.transaction_id
        for t in dataset.transactions
        if not math.isfinite(t.amount) or t.amount <= 0.0
    ]
    unrounded = [
        t.transaction_id
        for t in dataset.transactions
        if abs(t.amount - round(t.amount, 2)) > 1e-9
    ]

    assert not bad_value, f"non-positive or non-finite amounts on {bad_value[:5]}"
    assert not unrounded, f"amounts with more than two decimals on {unrounded[:5]}"


def test_categorical_fields_use_the_declared_vocabularies(dataset):
    assert {t.channel for t in dataset.transactions} <= set(CHANNELS)
    assert {t.status for t in dataset.transactions} <= set(STATUSES)
    assert {t.transaction_type for t in dataset.transactions} <= set(TRANSACTION_TYPES)
    assert {t.currency for t in dataset.transactions} == {dataset.config.currency}


def test_coordinates_are_on_the_globe(dataset):
    off_grid = [
        t.transaction_id
        for t in dataset.transactions
        if not (-90.0 <= t.lat <= 90.0 and -180.0 <= t.lon <= 180.0)
    ]

    assert not off_grid, f"coordinates out of range on {off_grid[:5]}"


def test_timestamps_are_timezone_aware_and_inside_the_observation_window(dataset):
    cfg = dataset.config
    end = dataset.boundaries.evaluation_end

    naive = [t.transaction_id for t in dataset.transactions if t.timestamp.tzinfo is None]
    assert not naive, f"naive timestamps on {naive[:5]}"

    outside = [
        t.transaction_id
        for t in dataset.transactions
        if not (cfg.warmup_start <= t.timestamp < end)
    ]
    assert not outside, (
        f"{len(outside)} rows outside [{cfg.warmup_start.isoformat()}, "
        f"{end.isoformat()}): {outside[:5]}"
    )


def test_label_columns_have_the_declared_types(dataset):
    """``is_fraud`` must be a real bool and ``attack_id`` a string or nothing."""
    for txn in dataset.transactions:
        assert isinstance(txn.is_fraud, bool)
        assert txn.attack_id is None or isinstance(txn.attack_id, str)
