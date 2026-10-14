"""Point-in-time history: the only sanctioned way to look at the past.

One rule, and it admits no variants:

    a feature for a transaction at ``as_of`` may read rows with ``ts < as_of``

Strictly less-than. A transaction never contributes to its own features, and
neither does any other row sharing its timestamp — two rows can land on the same
instant, and "the row is itself" is not a distinction a real system could make at
authorization time. Bounded windows are half-open backward, ``[as_of - Δ, as_of)``.
Centred and forward windows are not options that exist here: the only window
constructor is :func:`half_open_window`, and :meth:`HistoryView.rows` cannot be
asked for anything the cut does not already bound.

The mechanism is a hard index cut rather than a filter. :meth:`TransactionHistory.as_of`
resolves the as-of instant to an integer position once, and every reduction on the
resulting :class:`HistoryView` is bounded above by that position. There is no
predicate to get wrong, so a leak would require editing this module rather than
merely forgetting a condition — which is the whole reason
``docs/FEATURE_LEAKAGE_POLICY.md`` §2 insists features do not roll their own
windows.

Rows are stored as :func:`simulator.leakage.feature_safe_row` projections, so
``is_fraud`` and ``attack_id`` are *absent* from the store, not merely unread.

**Scope.** This is Checkpoint 1.5 safety infrastructure, not the Checkpoint 2
feature set. The reductions here are the primitives the five planned feature
families need — velocity counts, volume sums, historical amount deviation,
previous-transaction geography, and account age — and nothing more. The named
features of ``docs/FEATURE_REFERENCE.md`` (``tx_velocity_15m``,
``amount_deviation``, ``geo_distance_from_prev``, …) are built on top of these at
Checkpoint 2 and do not exist yet.

Every reduction is finite by construction: no reduction returns ``NaN`` or ``inf``
on empty or degenerate history, because a cold-start row must produce a neutral
value rather than an extreme one (``docs/FEATURE_REFERENCE.md`` §cold-start).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from simulator.base_generator import haversine_km
from simulator.leakage import feature_safe_row

__all__ = [
    "AmountStats",
    "ENTITY_KEYS",
    "HistoryView",
    "MIN_HISTORY_FOR_DEVIATION",
    "MIN_STD",
    "TransactionHistory",
    "half_open_window",
]

#: Columns a window may be grouped by. All are ``IDENTIFIER`` in
#: :mod:`ml.feature_contract`: legitimate for scoping history, never model inputs.
ENTITY_KEYS: Tuple[str, ...] = (
    "account_id",
    "customer_id",
    "device_id",
    "ip_id",
    "merchant_id",
    "beneficiary_id",
)

#: Below this many prior observations, a deviation score is noise dressed as
#: signal, so the neutral value 0.0 is returned instead.
MIN_HISTORY_FOR_DEVIATION: int = 5

#: Guard against dividing by a standard deviation of zero — constant history means
#: no statistical surprise, which is 0.0 and not infinity.
MIN_STD: float = 1e-6


def half_open_window(as_of: datetime, delta: timedelta) -> Tuple[datetime, datetime]:
    """The only sanctioned window: ``[as_of - delta, as_of)``.

    Returns ``(lower_inclusive, upper_exclusive)``. ``delta`` must be strictly
    positive; a zero or negative width would either be empty or reach forward, and
    both are refused loudly rather than silently returning nothing.
    """
    if delta <= timedelta(0):
        raise ValueError(
            f"window width must be strictly positive, got {delta!r}. "
            "Windows are backward-looking [as_of - delta, as_of); there is no "
            "forward or centred form."
        )
    return as_of - delta, as_of


@dataclass(frozen=True)
class AmountStats:
    """Guaranteed-finite summary of amounts over a window."""

    count: int
    mean: float
    std: float


class TransactionHistory:
    """Immutable, chronologically ordered store of feature-safe rows.

    Built once per dataset (or per split) and queried through :meth:`as_of`. The
    per-entity position index makes an entity-scoped window ``O(log n)`` in the
    number of that entity's rows rather than a scan of the whole store.

    Deliberately not a feature store: no registry, no caching layer, no
    persistence, no schema abstraction. It holds rows in a tuple and knows how to
    find the cut for an instant.
    """

    __slots__ = ("_rows", "_ts", "_index")

    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        ordered = sorted(rows, key=lambda r: (r["timestamp"], r["transaction_id"]))
        self._rows: Tuple[Mapping[str, Any], ...] = tuple(ordered)
        self._ts: Tuple[datetime, ...] = tuple(r["timestamp"] for r in self._rows)
        index: Dict[str, Dict[str, list]] = {key: {} for key in ENTITY_KEYS}
        for position, row in enumerate(self._rows):
            for key in ENTITY_KEYS:
                value = row.get(key)
                if value is not None:
                    index[key].setdefault(value, []).append(position)
        self._index: Mapping[str, Mapping[str, Tuple[int, ...]]] = {
            key: {v: tuple(p) for v, p in per_key.items()}
            for key, per_key in index.items()
        }

    @classmethod
    def from_transactions(cls, transactions: Iterable[Any]) -> "TransactionHistory":
        """Build from :class:`simulator.transactions.Transaction` objects.

        Each row is projected through :func:`simulator.leakage.feature_safe_row`, so
        the labels are not in the store at all. A feature cannot read what is not
        there, which is a stronger guarantee than a convention that it should not.
        """
        return cls([feature_safe_row(t) for t in transactions])

    def __len__(self) -> int:
        return len(self._rows)

    def as_of(self, instant: datetime) -> "HistoryView":
        """A view of everything strictly earlier than ``instant``.

        The cut is ``bisect_left``, which returns the first position whose timestamp
        is ``>= instant``. Positions ``[0, cut)`` are therefore all strictly
        earlier, including when several rows share ``instant`` exactly.
        """
        return HistoryView(self, bisect_left(self._ts, instant), instant)

    def view_for(self, txn: Any) -> "HistoryView":
        """The sanctioned view for scoring ``txn``: ``as_of = txn.timestamp``.

        The transaction is excluded from its own history even when it is present in
        the store, which is the normal case — history is built from the dataset the
        row belongs to.
        """
        return self.as_of(txn.timestamp)


class HistoryView:
    """A window onto history bounded above by one integer position.

    Holds no public handle to the parent store. Every reduction routes through
    :meth:`_bounds`, whose upper bound is the cut, so no method can reach a row at
    or after ``as_of``.
    """

    __slots__ = ("_history", "_cut", "as_of")

    def __init__(self, history: TransactionHistory, cut: int, as_of: datetime):
        self._history = history
        self._cut = cut
        #: The instant this view is anchored to. Rows at exactly this instant are out.
        self.as_of = as_of

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"HistoryView(as_of={self.as_of.isoformat()}, size={self._cut})"

    @property
    def size(self) -> int:
        """How many rows are strictly earlier than :attr:`as_of`."""
        return self._cut

    # -- window resolution -------------------------------------------------

    def _bounds(self, delta: Optional[timedelta]) -> Tuple[int, int]:
        """Resolve a window to positions ``[lo, hi)`` with ``hi`` capped at the cut.

        The single chokepoint. ``bisect_left`` is given ``hi=self._cut`` as its own
        upper search bound, so even the lower edge is computed without looking at a
        row the view may not see.
        """
        hi = self._cut
        if delta is None:
            return 0, hi
        lower, _upper = half_open_window(self.as_of, delta)
        return bisect_left(self._history._ts, lower, 0, hi), hi

    def _positions(
        self, delta: Optional[timedelta], key: Optional[str], value: Optional[str]
    ) -> Tuple[int, ...]:
        lo, hi = self._bounds(delta)
        if key is None:
            return tuple(range(lo, hi))
        if key not in ENTITY_KEYS:
            raise KeyError(f"{key!r} is not groupable; expected one of {ENTITY_KEYS}")
        if value is None:
            raise ValueError(f"grouping by {key!r} requires a value")
        entity_positions = self._history._index[key].get(value, ())
        start = bisect_left(entity_positions, lo)
        stop = bisect_left(entity_positions, hi)
        return entity_positions[start:stop]

    # -- reductions --------------------------------------------------------

    def rows(
        self,
        delta: Optional[timedelta] = None,
        *,
        key: Optional[str] = None,
        value: Optional[str] = None,
    ) -> Tuple[Mapping[str, Any], ...]:
        """Feature-safe rows in ``[as_of - delta, as_of)``, oldest first.

        ``delta=None`` means all available history, still strictly before
        :attr:`as_of`.
        """
        return tuple(self._history._rows[p] for p in self._positions(delta, key, value))

    def count(
        self,
        delta: Optional[timedelta] = None,
        *,
        key: Optional[str] = None,
        value: Optional[str] = None,
    ) -> int:
        """Row count — the primitive under every velocity feature."""
        return len(self._positions(delta, key, value))

    def amount_sum(
        self,
        delta: Optional[timedelta] = None,
        *,
        key: Optional[str] = None,
        value: Optional[str] = None,
    ) -> float:
        """Summed amount — the primitive under every transaction-volume feature."""
        rows = self._history._rows
        return float(sum(rows[p]["amount"] for p in self._positions(delta, key, value)))

    def amount_stats(
        self,
        delta: Optional[timedelta] = None,
        *,
        key: Optional[str] = None,
        value: Optional[str] = None,
    ) -> AmountStats:
        """Count, mean, and sample standard deviation of historical amounts.

        Empty history gives ``(0, 0.0, 0.0)`` and a single observation gives a zero
        standard deviation, both finite. Sample standard deviation (``ddof=1``)
        because the window is a sample of the entity's behaviour, not a population.
        """
        rows = self._history._rows
        amounts = [float(rows[p]["amount"]) for p in self._positions(delta, key, value)]
        n = len(amounts)
        if n == 0:
            return AmountStats(0, 0.0, 0.0)
        mean = math.fsum(amounts) / n
        if n == 1:
            return AmountStats(1, mean, 0.0)
        variance = math.fsum((a - mean) ** 2 for a in amounts) / (n - 1)
        return AmountStats(n, mean, math.sqrt(max(0.0, variance)))

    def deviation(
        self,
        amount: float,
        delta: Optional[timedelta] = None,
        *,
        key: Optional[str] = None,
        value: Optional[str] = None,
        min_history: int = MIN_HISTORY_FOR_DEVIATION,
        min_std: float = MIN_STD,
    ) -> float:
        """Guarded z-score of ``amount`` against historical amounts.

        Returns exactly ``0.0`` — "no deviation", the neutral claim — when history
        is too short or too constant to support a score, and is finite in every
        case. Manufacturing a z-score of 40 from two observations is worse than
        admitting there is no baseline yet.
        """
        stats = self.amount_stats(delta, key=key, value=value)
        if stats.count < min_history or stats.std <= min_std:
            return 0.0
        z = (float(amount) - stats.mean) / stats.std
        return z if math.isfinite(z) else 0.0

    def is_cold_start(
        self,
        delta: Optional[timedelta] = None,
        *,
        key: Optional[str] = None,
        value: Optional[str] = None,
        minimum: int = MIN_HISTORY_FOR_DEVIATION,
    ) -> bool:
        """Whether this window is too thin for a deviation score to mean anything."""
        return self.count(delta, key=key, value=value) < minimum

    def previous(
        self, *, key: str, value: str, delta: Optional[timedelta] = None
    ) -> Optional[Mapping[str, Any]]:
        """The most recent row before :attr:`as_of` for one entity, or ``None``.

        The primitive under the geographic previous-transaction features: the caller
        pairs it with :meth:`distance_from_km` to get a displacement, and gets
        ``None`` rather than a fabricated origin when there is no prior row.
        """
        positions = self._positions(delta, key, value)
        return self._history._rows[positions[-1]] if positions else None

    def distance_from_km(
        self, lat: float, lon: float, *, key: str, value: str
    ) -> Optional[float]:
        """Great-circle distance from the entity's previous transaction, in km.

        ``None`` when there is no previous transaction — deliberately not ``0.0``,
        which would claim "same place" and be indistinguishable from a genuine
        repeat at one location. Reuses
        :func:`simulator.base_generator.haversine_km` rather than restating the
        formula.
        """
        prior = self.previous(key=key, value=value)
        if prior is None:
            return None
        return haversine_km(float(prior["lat"]), float(prior["lon"]), float(lat), float(lon))

    def age_days(self, created_at: datetime) -> float:
        """Age at :attr:`as_of`, in days, clamped at zero.

        The primitive under account age. Clamped because a negative age would mean
        the entity was created after the transaction that references it — a
        causality violation the Checkpoint 1 validator already asserts is absent, so
        it must not silently become a large negative feature value here.
        """
        return max(0.0, (self.as_of - created_at).total_seconds() / 86400.0)
