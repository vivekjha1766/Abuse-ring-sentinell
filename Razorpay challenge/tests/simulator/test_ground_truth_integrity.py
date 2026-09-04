"""4. Ground-truth integrity — labels and the attack registry describe one story.

The registry is the answer key: every evaluation, every audit and every later
checkpoint's metrics are computed against it. Three things therefore have to hold
exactly, not approximately. Every fraudulent row must name an attack that exists.
Every normal row must name nothing at all — an ``attack_id`` on an unflagged row would
be a label visible to anything that joined on it. And the two directions must agree:
the set of rows the registry claims and the set of rows carrying ``is_fraud`` must be
the same set, with no row belonging to two attacks.

Record-level completeness is delegated to
:func:`simulator.ground_truth.validate_record_completeness` rather than restated here.
"""

from __future__ import annotations

from simulator.ground_truth import DETECTION_CLASS, validate_record_completeness


def test_every_record_is_complete(dataset):
    incomplete = {
        record.attack_id: list(problems)
        for record in dataset.ground_truth.records
        if (problems := validate_record_completeness(record))
    }

    assert not incomplete, f"incomplete ground-truth records: {list(incomplete)[:5]}"


def test_attack_ids_are_unique(dataset):
    ids = [r.attack_id for r in dataset.ground_truth.records]

    assert len(ids) == len(set(ids))


def test_flagged_rows_and_registry_agree(dataset):
    flagged = {t.transaction_id for t in dataset.transactions if t.is_fraud}
    claimed = dataset.ground_truth.fraudulent_transaction_ids()

    assert flagged - claimed == set(), "rows flagged is_fraud that no attack claims"
    assert claimed - flagged == set(), "rows claimed by an attack but not flagged"


def test_every_fraud_row_maps_to_a_valid_attack(dataset):
    registry = dataset.ground_truth
    known = {r.attack_id for r in registry.records}
    mapping = registry.transaction_to_attack()

    for txn in dataset.transactions:
        if not txn.is_fraud:
            continue
        assert txn.attack_id is not None, f"{txn.transaction_id} is fraud with no attack_id"
        assert txn.attack_id in known, f"{txn.transaction_id} names unknown {txn.attack_id}"
        assert mapping[txn.transaction_id] == txn.attack_id, (
            f"{txn.transaction_id} is claimed by {mapping[txn.transaction_id]} "
            f"but carries {txn.attack_id}"
        )
        # by_id raises KeyError on an unknown id, so this also exercises the lookup.
        assert registry.by_id(txn.attack_id).attack_id == txn.attack_id


def test_normal_rows_carry_no_attack_id(dataset):
    """Includes attack-generated cover traffic, which is normal by construction."""
    leaked = [
        t.transaction_id
        for t in dataset.transactions
        if not t.is_fraud and t.attack_id is not None
    ]

    assert not leaked, f"unflagged rows carrying an attack_id: {leaked[:5]}"


def test_no_row_is_claimed_by_two_attacks(dataset):
    seen: dict = {}
    for record in dataset.ground_truth.records:
        for txn_id in record.affected_transaction_ids:
            assert txn_id not in seen, (
                f"{txn_id} claimed by both {seen.get(txn_id)} and {record.attack_id}"
            )
            seen[txn_id] = record.attack_id


def test_record_counts_reconcile_with_the_rows(dataset):
    registry = dataset.ground_truth
    fraud_rows = sum(1 for t in dataset.transactions if t.is_fraud)

    assert sum(registry.family_counts().values()) == len(registry)
    assert sum(registry.transaction_counts_by_family().values()) == fraud_rows


def test_records_agree_with_the_split_and_window_of_their_rows(dataset):
    by_id = {t.transaction_id: t for t in dataset.transactions}

    for record in dataset.ground_truth.records:
        for txn_id in record.affected_transaction_ids:
            assert dataset.split_of[txn_id] == record.split, (
                f"{txn_id} sits in {dataset.split_of[txn_id]} but "
                f"{record.attack_id} declares {record.split}"
            )
            ts = by_id[txn_id].timestamp
            assert record.start_time <= ts <= record.end_time, (
                f"{txn_id} at {ts.isoformat()} is outside the declared window of "
                f"{record.attack_id}"
            )


def test_every_record_has_a_known_detection_class(dataset):
    """Class A (individual anomaly) and Class B (coordinated abuse) are reported
    separately throughout the project, so the partition must stay total."""
    classes = {r.detection_class for r in dataset.ground_truth.records}

    assert classes <= set(DETECTION_CLASS.values())
    assert {r.attack_type for r in dataset.ground_truth.records} <= set(DETECTION_CLASS)
