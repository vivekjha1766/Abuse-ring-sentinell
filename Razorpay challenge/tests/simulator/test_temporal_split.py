"""5. Temporal split integrity — warm-up, train, validation, held-out.

The split is the load-bearing wall of every honest evaluation in this project: the
held-out window must never be tuned against, and a feature computed in one split must
never see a row from a later one. That guarantee reduces to three mechanical facts,
which this file pins.

*Coverage*: every transaction is labelled with exactly one split, and the label agrees
with what :meth:`SplitBoundaries.split_for` would say about its timestamp — so the
recorded ``split_of`` map cannot drift away from the boundaries it claims to encode.

*Containment*: each row's timestamp lies inside its own split's half-open window
``[start, end)``. Half-open matters — a row landing exactly on a boundary belongs to
the *later* split, never both and never neither.

*Ordering with no overlap*: the whole of train precedes the whole of validation, which
precedes the whole of held-out, compared as observed extremes rather than as declared
boundaries. Comparing declared boundaries would only re-test
``SplitBoundaries.__post_init__``; comparing the rows tests the assignment.

Split *proportions* are deliberately not asserted. They are quantile-derived from
normal traffic before attacks are injected, so the realised shares move with the attack
mix; ``validate_dataset`` already reports drift there at warning level, and pinning a
tolerance here would only add a brittle second opinion.
"""

from __future__ import annotations

import pytest

from simulator.config import SPLIT_NAMES
from simulator.pipeline import WARMUP_SPLIT

ALL_SPLITS = (WARMUP_SPLIT,) + tuple(SPLIT_NAMES)


def test_every_transaction_has_exactly_one_known_split(dataset):
    ids = {t.transaction_id for t in dataset.transactions}

    assert set(dataset.split_of) == ids, "split_of does not cover exactly the rows"
    assert set(dataset.split_of.values()) <= set(ALL_SPLITS)


def test_recorded_split_agrees_with_the_boundaries(dataset):
    """``split_of`` is a cache of :meth:`split_for`; the two must not diverge."""
    disagreements = [
        (txn.transaction_id, dataset.split_of[txn.transaction_id], expected)
        for txn in dataset.transactions
        if (expected := dataset.boundaries.split_for(txn.timestamp))
        != dataset.split_of[txn.transaction_id]
    ]

    assert not disagreements, f"recorded split differs from split_for: {disagreements[:5]}"


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_no_row_falls_outside_its_declared_window(dataset, split):
    window = dataset.boundaries.window(split)

    outside = [
        (txn.transaction_id, txn.timestamp.isoformat())
        for txn in dataset.transactions
        if dataset.split_of[txn.transaction_id] == split
        and not (window.start <= txn.timestamp < window.end)
    ]

    assert not outside, (
        f"{len(outside)} {split} rows outside [{window.start.isoformat()}, "
        f"{window.end.isoformat()}): {outside[:5]}"
    )


def test_warmup_rows_precede_the_evaluation_window(dataset):
    start = dataset.boundaries.evaluation_start

    late = [
        txn.transaction_id
        for txn in dataset.transactions
        if dataset.split_of[txn.transaction_id] == WARMUP_SPLIT
        and txn.timestamp >= start
    ]

    assert not late, f"warm-up rows at or after {start.isoformat()}: {late[:5]}"


def test_splits_are_strictly_ordered_and_do_not_overlap(dataset):
    """Observed extremes, not declared boundaries — this tests the assignment."""
    minima = {}
    maxima = {}
    for txn in dataset.transactions:
        split = dataset.split_of[txn.transaction_id]
        ts = txn.timestamp
        if split not in minima or ts < minima[split]:
            minima[split] = ts
        if split not in maxima or ts > maxima[split]:
            maxima[split] = ts

    for split in SPLIT_NAMES:
        assert split in minima, f"split {split!r} received no transactions"

    assert maxima["train"] < minima["validation"], "train overlaps validation"
    assert maxima["validation"] < minima["heldout"], "validation overlaps held-out"
    assert minima["train"] >= dataset.boundaries.evaluation_start
    assert maxima["heldout"] < dataset.boundaries.evaluation_end


def test_warmup_carries_no_fraud_and_no_attacks(dataset):
    """Warm-up exists to give point-in-time features history, not signal.

    Fraud there would be unlabelled-in-practice history that a feature could still
    aggregate over, so the window is kept clean of attacks entirely.
    """
    fraud = [
        txn.transaction_id
        for txn in dataset.transactions
        if dataset.split_of[txn.transaction_id] == WARMUP_SPLIT and txn.is_fraud
    ]

    assert not fraud, f"fraudulent rows inside the warm-up window: {fraud[:5]}"
    assert dataset.ground_truth.by_split(WARMUP_SPLIT) == []


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_attack_windows_sit_inside_their_split_window(dataset, split):
    """A record must not straddle a boundary, or its rows would span two splits."""
    window = dataset.boundaries.window(split)

    straddling = [
        record.attack_id
        for record in dataset.ground_truth.by_split(split)
        if not (window.start <= record.start_time and record.end_time < window.end)
    ]

    assert not straddling, f"{split} attacks crossing their window: {straddling[:5]}"
