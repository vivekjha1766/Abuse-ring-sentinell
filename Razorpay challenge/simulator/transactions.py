"""Transaction model and the normal-behaviour generator.

The :class:`Transaction` dataclass matches ``DATA_DICTIONARY.md`` §3 field for
field. Two of those fields — ``is_fraud`` and ``attack_id`` — are ground-truth
labels. They are physically present on the row but are **evaluation-only** and
must never be read by feature, model, or scoring code
(``docs/FEATURE_LEAKAGE_POLICY.md`` §4).

The normal generator is where the dataset earns its credibility. If normal
behaviour were simplistic, fraud would be trivially separable and every metric
downstream would be meaningless. So normal traffic here is deliberately
non-trivial (PRD FR-1):

* heterogeneous merchant volumes (popularity-weighted choice),
* per-account spending distributions (log-normal around a personal mean),
* genuinely large legitimate outliers,
* device/IP selection that reuses each account's *legitimate* pool, including
  shared household devices and NAT/CGNAT IPs,
* occasional legitimate travel, producing benign geographic spread,
* hour-of-day and day-of-week seasonality via thinning.

The guiding invariant, checked later in ``docs/ML_READINESS.md`` §3: neither
"large amount" nor "high count" may by itself separate fraud from normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from simulator.base_generator import (
    SeedRegistry,
    clamp,
    offset_coordinates,
    round_amount,
    sample_seasonal_timestamps,
)
from simulator.entities import Account, Ecosystem

__all__ = ["Transaction", "generate_normal_transactions", "TRANSACTION_FIELDS"]

CHANNELS = ("web", "mobile", "pos", "api")
TRANSACTION_TYPES = ("purchase", "transfer", "withdrawal", "refund")
STATUSES = ("completed", "pending", "failed")


@dataclass(frozen=True)
class Transaction:
    """One payment event (DATA_DICTIONARY.md §3)."""

    transaction_id: str
    timestamp: datetime
    account_id: str
    customer_id: str
    device_id: str
    ip_id: str
    amount: float
    currency: str
    channel: str
    transaction_type: str
    lat: float
    lon: float
    status: str
    merchant_id: Optional[str] = None
    beneficiary_id: Optional[str] = None
    # -- ground-truth labels: evaluation only, quarantined from all inference --
    is_fraud: bool = False
    attack_id: Optional[str] = None


#: Canonical column order for tabular export. Ground-truth columns are last so
#: they are easy to drop as a block when building the feature matrix.
TRANSACTION_FIELDS: tuple[str, ...] = (
    "transaction_id", "timestamp", "account_id", "customer_id", "merchant_id",
    "beneficiary_id", "device_id", "ip_id", "amount", "currency", "channel",
    "transaction_type", "lat", "lon", "status", "is_fraud", "attack_id",
)


def _account_status_active(account: Account) -> bool:
    return account.status == "active"


def _sample_amount(rng: np.random.Generator, account: Account) -> float:
    """Log-normal amount around the account's personal mean.

    Parameterised so the *median* sits near ``baseline_amount_mean`` and the
    spread follows the account's own volatility. This gives every account its
    own distribution — a premium account's ordinary purchase can exceed a budget
    account's largest, which is exactly why a global amount threshold fails.
    """
    mean = max(1.0, account.baseline_amount_mean)
    std = max(0.5, account.baseline_amount_std)
    cv = clamp(std / mean, 0.05, 3.0)
    sigma = float(np.sqrt(np.log1p(cv * cv)))
    mu = float(np.log(mean) - 0.5 * sigma * sigma)
    return round_amount(float(rng.lognormal(mu, sigma)))


def _choose_merchant(rng: np.random.Generator, eco: Ecosystem, ts: datetime) -> str:
    """Popularity-weighted merchant choice, restricted to merchants that exist.

    Merchants are onboarded throughout the timeline now, so a transaction must not
    be attributed to a merchant that has not been created yet — that would be
    future information leaking into a historical row, which
    ``docs/FEATURE_LEAKAGE_POLICY.md`` forbids outright. Rejection sampling keeps
    the popularity weighting exactly intact (a filtered-and-renormalised
    distribution would not), and the rejection rate is low because only
    ``new_entity_fraction`` of merchants are joiners and each is unusable for only
    the part of the window that precedes it.
    """
    ids = eco._merchant_ids_cache  # type: ignore[attr-defined]
    weights = eco._merchant_weights_cache  # type: ignore[attr-defined]
    created = eco._merchant_created_cache  # type: ignore[attr-defined]
    for _ in range(12):
        idx = int(rng.choice(len(ids), p=weights))
        if created[idx] <= ts:
            return ids[idx]
    # Fall back to merchants that predate the whole dataset, which every timestamp
    # can reach. Reached only when the draws keep landing on future joiners.
    old_ids = eco._merchant_old_ids_cache  # type: ignore[attr-defined]
    old_weights = eco._merchant_old_weights_cache  # type: ignore[attr-defined]
    return old_ids[int(rng.choice(len(old_ids), p=old_weights))]


def _select_device_ip(
    rng: np.random.Generator,
    eco: Ecosystem,
    account: Account,
    ts: datetime,
    device_created: Dict[str, datetime],
    ip_created: Dict[str, datetime],
    switch_dev: float,
    switch_ip: float,
) -> tuple[str, str]:
    """Pick the device and IP this account plausibly used at ``ts``.

    Two rules, both about causality. An account cannot use hardware or an address
    that does not exist yet, and once it has *adopted* a new device (bought a new
    phone) that device becomes its primary rather than an occasional alternate —
    otherwise mid-timeline devices would appear a handful of times each and carry
    no real volume, leaving "unseen device" still effectively a fraud marker.
    """
    aid = account.account_id
    devices = eco.account_devices.get(aid) or [eco.devices[0].device_id]
    ips = eco.account_ips.get(aid) or [eco.ips[0].ip_id]

    usable_devices = [d for d in devices if device_created.get(d, ts) <= ts]
    if not usable_devices:
        usable_devices = [min(devices, key=lambda d: device_created.get(d, ts))]
    usable_ips = [i for i in ips if ip_created.get(i, ts) <= ts]
    if not usable_ips:
        usable_ips = [min(ips, key=lambda i: ip_created.get(i, ts))]

    adopted = eco.account_device_adoption.get(aid)
    if adopted is not None and ts >= adopted[1] and adopted[0] in usable_devices:
        primary_device = adopted[0]
    else:
        primary_device = usable_devices[0]
    rotated = eco.account_ip_adoption.get(aid)
    if rotated is not None and ts >= rotated[1] and rotated[0] in usable_ips:
        primary_ip = rotated[0]
    else:
        primary_ip = usable_ips[0]

    if len(usable_devices) > 1 and rng.random() < switch_dev:
        device_id = usable_devices[int(rng.integers(0, len(usable_devices)))]
    else:
        device_id = primary_device
    if len(usable_ips) > 1 and rng.random() < switch_ip:
        ip_id = usable_ips[int(rng.integers(0, len(usable_ips)))]
    else:
        ip_id = primary_ip
    return device_id, ip_id


def _customer_trips(
    rng: np.random.Generator,
    cfg,
    customer,
    span_start: datetime,
    span_end: datetime,
) -> List[tuple]:
    """Coherent travel episodes for one customer: ``(start_s, end_s, lat, lon)``.

    Trips rather than independent per-transaction jitter, because the two are not
    equivalent from a detector's point of view. Isolated far-away transactions
    leave a *run* of consecutive distant transactions unique to account takeover,
    which is exactly the shortcut being closed. A real trip moves everything the
    person does for days at a time.
    """
    span = (span_end - span_start).total_seconds()
    if span <= 0:
        return []
    years = span / (365.25 * 86400.0)
    n_trips = min(8, int(rng.poisson(max(0.0, cfg.trips_per_year * years))))
    base = span_start.timestamp()
    trips: List[tuple] = []
    for _ in range(n_trips):
        start_s = base + float(rng.random()) * span
        dur = float(rng.uniform(*cfg.trip_duration_days)) * 86400.0
        if float(rng.random()) < cfg.long_haul_share:
            dist = float(rng.uniform(*cfg.long_haul_range))
        else:
            dist = float(rng.uniform(*cfg.trip_distance_range))
        lat, lon = offset_coordinates(
            customer.home_lat, customer.home_lon, dist, float(rng.random()) * 2 * np.pi
        )
        trips.append((start_s, start_s + dur, lat, lon))
    return trips


def generate_normal_transactions(eco: Ecosystem, seeds: SeedRegistry) -> List[Transaction]:
    """Generate all legitimate transactions across warm-up + evaluation window.

    The warm-up period (``config.warmup_days``) is normal-only, giving 7-day
    features full history by the time the evaluation window opens
    (``docs/DATA_SPLIT_PROTOCOL.md`` §5).

    Every row respects causality: an account transacts only after it is opened, and
    only through devices, IPs, merchants and beneficiaries that already exist at
    that instant. Before this was enforced the audit measured negative
    ``account_age_days`` and ``customer_age_days`` values — transactions that
    preceded their own account — which is future information in the crudest form.
    """
    rng = seeds.rng("transactions/normal")
    cfg = eco.config
    eco_cfg = cfg.ecosystem
    start = cfg.warmup_start
    end = cfg.end

    # Cache merchant sampling arrays once (popularity-weighted, heterogeneous).
    merchant_ids = [m.merchant_id for m in eco.merchants]
    weights = np.array([eco.merchant_weight[m] for m in merchant_ids], dtype=float)
    weights = weights / weights.sum()
    merchant_created = [m.created_at for m in eco.merchants]
    old_pairs = [
        (m.merchant_id, eco.merchant_weight[m.merchant_id])
        for m in eco.merchants
        if m.created_at < start
    ] or [(m.merchant_id, eco.merchant_weight[m.merchant_id]) for m in eco.merchants]
    old_ids = [p[0] for p in old_pairs]
    old_weights = np.array([p[1] for p in old_pairs], dtype=float)
    old_weights = old_weights / old_weights.sum()
    eco._merchant_ids_cache = merchant_ids  # type: ignore[attr-defined]
    eco._merchant_weights_cache = weights  # type: ignore[attr-defined]
    eco._merchant_created_cache = merchant_created  # type: ignore[attr-defined]
    eco._merchant_old_ids_cache = old_ids  # type: ignore[attr-defined]
    eco._merchant_old_weights_cache = old_weights  # type: ignore[attr-defined]

    merchant_loc = {m.merchant_id: (m.lat, m.lon) for m in eco.merchants}
    device_created = {d.device_id: d.created_at for d in eco.devices}
    ip_created = {i.ip_id: i.created_at for i in eco.ips}

    beneficiaries = [(b.beneficiary_id, b.created_at) for b in eco.beneficiaries]
    old_beneficiary_ids = [b for b, c in beneficiaries if c < start] or [
        b for b, _c in beneficiaries
    ]
    beneficiary_created = {b.beneficiary_id: b.created_at for b in eco.beneficiaries}

    def pick_beneficiary(account_id: str, ts: datetime) -> Optional[str]:
        """A payee from *this account's* list, not from the global pool.

        Drawing globally made every legitimate account accumulate as many distinct
        payees as it made transfers, so payee concentration became a near-perfect
        fraud rule — see
        :attr:`simulator.config.EcosystemConfig.beneficiaries_per_account` for the
        measured AUCs. The account's own list is filtered by creation time, because a
        payee registered later in the timeline cannot be paid now.
        """
        if not beneficiaries:
            return None
        payees = [
            b
            for b in eco.account_beneficiaries.get(account_id, ())
            if beneficiary_created.get(b, ts) <= ts
        ]
        if payees:
            return payees[int(rng.integers(0, len(payees)))]
        # An account whose whole list postdates ``ts`` still needs somewhere to pay;
        # fall back to a payee that predates the dataset, which every row can reach.
        return old_beneficiary_ids[int(rng.integers(0, len(old_beneficiary_ids)))]

    trips_cache: Dict[str, List[tuple]] = {}
    transactions: List[Transaction] = []
    for account in eco.accounts:
        if account.status == "closed":
            continue
        # An account cannot transact before it is opened. Accounts opened partway
        # through therefore have genuinely short histories and low counts, which is
        # what stops a low transaction count from meaning "synthetic identity".
        active_start = max(start, account.created_at)
        active_days = (end - active_start).total_seconds() / 86400.0
        if active_days <= 0:
            continue
        freq = account.baseline_tx_frequency
        if account.status == "dormant":
            freq *= 0.15
        expected = freq * active_days
        # Personal activity varies run to run but stays anchored to the baseline.
        n_tx = int(rng.poisson(max(0.0, expected)))
        if n_tx <= 0:
            continue

        customer = eco.customer_by_id[account.customer_id]
        timestamps = sample_seasonal_timestamps(
            rng, n_tx, active_start, end, eco_cfg.hour_weights, eco_cfg.dow_weights
        )
        trips = trips_cache.get(customer.customer_id)
        if trips is None:
            trips = _customer_trips(rng, eco_cfg, customer, start, end)
            trips_cache[customer.customer_id] = trips

        for ts in timestamps:
            is_transfer = rng.random() < eco_cfg.transfer_fraction
            amount = _sample_amount(rng, account)
            # Legitimate high-value outlier: rare, large, and genuinely normal.
            if rng.random() < eco_cfg.legit_outlier_rate:
                lo, hi = eco_cfg.legit_outlier_scale
                amount = round_amount(amount * float(rng.uniform(lo, hi)))
            # Legitimate *small* purchase, independent of the account's own mean:
            # a coffee, a fare, a tip. Drawn log-uniformly so the low decade is
            # populated evenly rather than clustering at the top of the range.
            # This is what stops "amount is tiny" from meaning "microtransaction
            # attack" (``docs/ML_READINESS.md`` §3).
            elif rng.random() < eco_cfg.micro_spend_rate:
                lo, hi = eco_cfg.micro_spend_range
                amount = round_amount(
                    float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                )
            # Legitimate *large* transfer: rent, tuition, an invoice, moving money
            # between one's own accounts. Keeps four-figure transfers from being a
            # circular-money-flow marker.
            if is_transfer and rng.random() < eco_cfg.large_transfer_prob:
                lo, hi = eco_cfg.large_transfer_range
                amount = round_amount(
                    float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                )

            device_id, ip_id = _select_device_ip(
                rng, eco, account, ts, device_created, ip_created,
                eco_cfg.device_switch_prob, eco_cfg.ip_switch_prob,
            )

            merchant_id: Optional[str] = None
            beneficiary_id: Optional[str] = None
            if is_transfer and beneficiaries:
                tx_type = "transfer"
                beneficiary_id = pick_beneficiary(account.account_id, ts)
                channel = "web" if rng.random() < 0.6 else "mobile"
            else:
                r = rng.random()
                tx_type = "purchase" if r < 0.9 else ("withdrawal" if r < 0.97 else "refund")
                merchant_id = _choose_merchant(rng, eco, ts)
                channel = CHANNELS[int(rng.choice(len(CHANNELS), p=[0.34, 0.42, 0.18, 0.06]))]

            # --- location -------------------------------------------------
            # Card-not-present purchases resolve to the *merchant*, which is how
            # online payments actually geolocate and which spreads the distance
            # distribution into a continuous tail instead of a spike at the
            # cardholder's doorstep.
            ts_epoch = ts.timestamp()
            on_trip = next(
                ((la, lo) for s, e, la, lo in trips if s <= ts_epoch < e), None
            )
            if (
                merchant_id is not None
                and channel in ("web", "api")
                and rng.random() < eco_cfg.merchant_geolocated_share
            ):
                m_lat, m_lon = merchant_loc[merchant_id]
                lat, lon = offset_coordinates(
                    m_lat, m_lon, float(rng.random()) * 25.0,
                    float(rng.random()) * 2 * np.pi,
                )
            elif on_trip is not None:
                lat, lon = offset_coordinates(
                    on_trip[0], on_trip[1], float(rng.random()) * 30.0,
                    float(rng.random()) * 2 * np.pi,
                )
            elif rng.random() < eco_cfg.travel_prob:
                dist = float(rng.uniform(30.0, 900.0))
                lat, lon = offset_coordinates(
                    customer.home_lat, customer.home_lon, dist, float(rng.random()) * 2 * np.pi
                )
            else:
                lat, lon = offset_coordinates(
                    customer.home_lat, customer.home_lon, float(rng.random()) * 8.0,
                    float(rng.random()) * 2 * np.pi,
                )

            u = rng.random()
            status = "completed" if u < 0.965 else ("pending" if u < 0.99 else "failed")

            transactions.append(
                Transaction(
                    transaction_id=eco.new_id("transaction", "txn"),
                    timestamp=ts,
                    account_id=account.account_id,
                    customer_id=account.customer_id,
                    merchant_id=merchant_id,
                    beneficiary_id=beneficiary_id,
                    device_id=device_id,
                    ip_id=ip_id,
                    amount=amount,
                    currency=cfg.currency,
                    channel=channel,
                    transaction_type=tx_type,
                    lat=lat,
                    lon=lon,
                    status=status,
                    is_fraud=False,
                    attack_id=None,
                )
            )

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    return transactions


def summarize_amounts(transactions: List[Transaction]) -> Dict[str, float]:
    """Quick amount summary, used by validation / readiness smoke checks."""
    amounts = np.array([t.amount for t in transactions], dtype=float)
    if amounts.size == 0:
        return {"count": 0.0}
    return {
        "count": float(amounts.size),
        "mean": float(amounts.mean()),
        "median": float(np.median(amounts)),
        "p99": float(np.percentile(amounts, 99)),
        "max": float(amounts.max()),
    }
