"""Point-in-time safety: a future row cannot reach backwards.

``docs/FEATURE_LEAKAGE_POLICY.md`` §2 states the rule as ``ts < as_of`` with
half-open backward windows. These tests attack the four ways that rule actually
breaks in practice:

*Contamination by a later row.* Features are computed, later transactions are then
appended, and the features are recomputed. Any difference is leakage.

*Contamination at the boundary.* The row being scored is in the store — that is the
normal case — and so are rows sharing its exact timestamp. All must be excluded,
because ``<=`` instead of ``<`` is the single most likely implementation slip.

*A window that reaches forward.* The only window constructor refuses a
non-positive width, so a centred or forward window cannot be spelled.

*A cold start that poisons the model.* Empty or constant history must produce a
finite neutral value; ``NaN`` and ``inf`` propagate silently through training and
surface as an unexplained failure much later.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from ml.point_in_time import (
    MIN_HISTORY_FOR_DEVIATION,
    TransactionHistory,
    half_open_window,
)

#: ``docs/FEATURE_LEAKAGE_POLICY.md`` §4 — these guards must never be skipped.
pytestmark = pytest.mark.leakage

HOUR = timedelta(hours=1)


def _signature(view, txn) -> dict:
    """Every reduction the view offers, in one comparable dict.

    Comparing a signature before and after injecting future rows tests the whole
    surface at once, so a new reduction added without a bound is caught by the
    existing tests rather than needing its own.
    """
    key, value = "account_id", txn.account_id
    previous = view.previous(key=key, value=value)
    return {
        "size": view.size,
        "count_all": view.count(),
        "count_scoped": view.count(key=key, value=value),
        "count_1h": view.count(HOUR, key=key, value=value),
        "sum_all": view.amount_sum(),
        "sum_1h": view.amount_sum(HOUR, key=key, value=value),
        "stats": view.amount_stats(key=key, value=value),
        "deviation": view.deviation(txn.amount, key=key, value=value),
        "cold_start": view.is_cold_start(key=key, value=value),
        "previous_id": previous["transaction_id"] if previous else None,
        "distance_km": view.distance_from_km(txn.lat, txn.lon, key=key, value=value),
        "row_ids": tuple(r["transaction_id"] for r in view.rows(key=key, value=value)),
    }


def test_future_rows_cannot_change_any_reduction(make_txn):
    """The headline guarantee: appending later traffic changes nothing."""
    past = [make_txn(i, minutes=i * 10, amount=100.0 + i) for i in range(8)]
    target = make_txn(99, minutes=100, amount=5000.0)

    before = _signature(TransactionHistory.from_transactions(past).view_for(target), target)

    future = [make_txn(500 + i, minutes=200 + i * 10, amount=9_000.0 + i) for i in range(20)]
    after = _signature(
        TransactionHistory.from_transactions(past + [target] + future).view_for(target),
        target,
    )

    assert before == after


def test_future_amount_cannot_move_a_historical_deviation(make_txn):
    """A large later amount must not shift the mean it is compared against."""
    past = [make_txn(i, minutes=i * 10, amount=100.0) for i in range(MIN_HISTORY_FOR_DEVIATION + 2)]
    target = make_txn(99, minutes=500, amount=140.0)

    quiet = TransactionHistory.from_transactions(past)
    loud = TransactionHistory.from_transactions(
        past + [make_txn(900, minutes=900, amount=1_000_000.0)]
    )

    assert quiet.view_for(target).amount_stats() == loud.view_for(target).amount_stats()
    assert quiet.view_for(target).deviation(target.amount) == loud.view_for(target).deviation(
        target.amount
    )


def test_a_transaction_is_excluded_from_its_own_history(make_txn):
    """The row being scored is in the store and must not see itself."""
    rows = [make_txn(i, minutes=i * 10) for i in range(6)]
    target = rows[3]

    view = TransactionHistory.from_transactions(rows).view_for(target)

    assert target.transaction_id not in {r["transaction_id"] for r in view.rows()}
    assert view.size == 3


def test_rows_sharing_the_as_of_instant_are_excluded(make_txn):
    """Strictly-less-than, not less-than-or-equal: ties are the future too."""
    simultaneous = [make_txn(i, minutes=60, amount=10.0 * i) for i in range(4)]
    earlier = make_txn(90, minutes=30)
    target = simultaneous[1]

    view = TransactionHistory.from_transactions([earlier, *simultaneous]).view_for(target)

    assert view.size == 1
    assert view.rows()[0]["transaction_id"] == earlier.transaction_id


def test_window_lower_edge_included_upper_edge_excluded(make_txn):
    """``[as_of - Δ, as_of)`` — one boundary is in, the other is out."""
    target = make_txn(99, minutes=120)
    on_lower_edge = make_txn(1, minutes=60)  # exactly as_of - 1h
    just_inside = make_txn(2, minutes=61)
    on_upper_edge = make_txn(3, minutes=120)  # exactly as_of
    just_outside = make_txn(4, minutes=59)

    view = TransactionHistory.from_transactions(
        [on_lower_edge, just_inside, on_upper_edge, just_outside]
    ).view_for(target)

    assert {r["transaction_id"] for r in view.rows(HOUR)} == {
        on_lower_edge.transaction_id,
        just_inside.transaction_id,
    }


def test_no_forward_or_centred_window_can_be_constructed(origin):
    """The only constructor refuses every width that is not backward-looking."""
    lower, upper = half_open_window(origin, HOUR)
    assert (lower, upper) == (origin - HOUR, origin)

    for bad_width in (timedelta(0), timedelta(hours=-1)):
        with pytest.raises(ValueError):
            half_open_window(origin, bad_width)


def test_no_reduction_can_reach_past_the_cut(make_txn):
    """Whatever is asked of the view, every row it returns is strictly earlier."""
    rows = [make_txn(i, minutes=i * 5, account=f"acc_{i % 3}") for i in range(40)]
    target = rows[20]
    view = TransactionHistory.from_transactions(rows).view_for(target)

    assert view.size < len(rows)
    for delta in (None, HOUR, timedelta(days=7)):
        for key, value in (("account_id", target.account_id), (None, None)):
            for row in view.rows(delta, key=key, value=value):
                assert row["timestamp"] < view.as_of


def test_entity_scoping_does_not_mix_entities(make_txn):
    """A per-account velocity must not count another account's rows."""
    mine = [make_txn(i, minutes=i, account="acc_1") for i in range(5)]
    theirs = [make_txn(100 + i, minutes=i, account="acc_2") for i in range(30)]
    target = make_txn(999, minutes=50, account="acc_1")

    view = TransactionHistory.from_transactions(mine + theirs).view_for(target)

    assert view.count(key="account_id", value="acc_1") == 5
    assert view.count(key="account_id", value="acc_2") == 30
    assert view.count() == 35


def test_recomputation_is_deterministic_regardless_of_input_order(make_txn):
    """Same rows, different insertion order, identical features."""
    rows = [make_txn(i, minutes=i * 7, amount=50.0 + i * 3) for i in range(12)]
    target = rows[9]

    forward = _signature(TransactionHistory.from_transactions(rows).view_for(target), target)
    reversed_ = _signature(
        TransactionHistory.from_transactions(list(reversed(rows))).view_for(target), target
    )

    assert forward == reversed_


def test_reductions_stay_finite_on_empty_and_constant_history(make_txn):
    """Cold start yields a neutral value, never ``NaN``, ``inf``, or a crash."""
    target = make_txn(99, minutes=100, amount=250.0)

    empty = TransactionHistory.from_transactions([]).view_for(target)
    assert empty.size == 0
    assert empty.count() == 0
    assert empty.amount_sum() == 0.0
    assert empty.amount_stats().count == 0
    assert empty.deviation(target.amount) == 0.0
    assert empty.previous(key="account_id", value=target.account_id) is None
    assert empty.distance_from_km(1.0, 2.0, key="account_id", value=target.account_id) is None
    assert empty.is_cold_start()

    constant = TransactionHistory.from_transactions(
        [make_txn(i, minutes=i, amount=100.0) for i in range(20)]
    ).view_for(target)
    assert constant.amount_stats().std == 0.0
    assert constant.deviation(target.amount) == 0.0

    for value in (
        empty.deviation(target.amount),
        constant.deviation(target.amount),
        empty.amount_sum(),
        constant.age_days(target.timestamp),
    ):
        assert math.isfinite(value)


def test_the_history_store_does_not_contain_the_labels(make_txn):
    """Quarantine at the storage layer: absent, not merely unread."""
    rows = [make_txn(i, minutes=i, is_fraud=(i % 2 == 0)) for i in range(6)]
    view = TransactionHistory.from_transactions(rows).view_for(make_txn(99, minutes=100))

    for row in view.rows():
        assert "is_fraud" not in row
        assert "attack_id" not in row
        with pytest.raises(KeyError):
            row["is_fraud"]


def test_age_days_is_never_negative(make_txn, origin):
    """A creation date after ``as_of`` clamps to zero instead of going negative."""
    view = TransactionHistory.from_transactions([]).view_for(make_txn(1, minutes=0))

    assert view.age_days(origin - timedelta(days=30)) == pytest.approx(30.0)
    assert view.age_days(origin + timedelta(days=30)) == 0.0
