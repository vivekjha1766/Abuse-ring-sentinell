"""Attack family framework.

Every attack family implements the same small contract: given an
:class:`AttackContext` and its own typed parameter object, produce an
:class:`AttackResult` holding the injected transactions plus the ground-truth
record describing them.

Two design rules apply to every family.

**No hardcoded shapes** (PRD FR-2). Sizes, durations, amounts, intensities and
counts all come from the parameter object. A family reads its params and nothing
else, so changing a parameter measurably changes the data and two runs with the
same seed and params are identical.

**Rings must be embedded, not bolted on** (``docs/ML_READINESS.md`` §8). Attacks
recruit *existing* legitimate accounts wherever the scenario allows, rather than
inventing a private population. A ring of pre-existing accounts still carries
its members' ordinary traffic, ordinary devices and ordinary merchants, so the
ring does not appear as a disconnected component that a trivial connectivity
filter would isolate. Where a family must create entities to be realistic — fake
merchants are genuinely newly onboarded, synthetic identities are by definition
new — those entities are wired into existing infrastructure and existing
merchants so they remain part of the main graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from simulator.base_generator import TimeWindow, offset_coordinates, round_amount
from simulator.entities import Account, Ecosystem
from simulator.ground_truth import AttackRecord
from simulator.transactions import CHANNELS, Transaction

__all__ = [
    "AttackContext",
    "AttackResult",
    "EntityCollector",
    "attack_window",
    "account_devices_ips",
    "account_home",
    "capped_entities",
    "emit",
    "emit_cover",
    "entity_created_at",
    "jittered_amount",
    "merchant_sampler",
    "pick_accounts",
    "sample_channel",
    "sample_status",
    "severity_for",
    "spread_timestamps",
    "vary_count",
]


@dataclass
class AttackContext:
    """Everything an attack family needs, and nothing more."""

    eco: Ecosystem
    rng: np.random.Generator
    split: str
    """Temporal split this attack is injected into."""
    window: TimeWindow
    """The split's time range. The attack must live entirely inside it, so that
    split assignment by timestamp is unambiguous."""
    attack_id: str
    account_activity: Dict[str, int] = field(default_factory=dict)
    """account_id -> count of that account's existing normal transactions.
    Lets families that need an established victim (takeover, spike) pick one."""
    merchant_activity: Dict[str, int] = field(default_factory=dict)
    """merchant_id -> count of that merchant's existing organic purchases.

    Lets ``fake_merchant_ring`` check whether a shell storefront's own trading
    history is already thick enough that its ring rows stay a small minority of it,
    and inject top-up cover traffic only for the shortfall. Reading *volume* is not
    reading a label: this is the same count any acquirer sees when it reviews a
    merchant, and it is derived from normal traffic alone."""
    merchant_fraud_rows: Dict[str, int] = field(default_factory=dict)
    """merchant_id -> fraud rows already placed on it *in this split*.

    Shared across every attack instance in the split and updated in place, so a
    family that recycles a merchant can see the exposure earlier instances gave it.
    This is generator bookkeeping, not an observable: the injector writes it and the
    injector reads it, and it never reaches a :class:`Transaction`."""
    exclude_accounts: set = field(default_factory=set)
    """account_ids already claimed by earlier attacks in this split. Recruiting
    is still allowed to overlap in principle, but the pipeline uses this to keep
    each account attributable to at most one attack, so ground-truth attribution
    stays unambiguous."""

    # -- convenience ------------------------------------------------------
    @property
    def currency(self) -> str:
        return self.eco.config.currency

    def new_transaction_id(self) -> str:
        """Mint an ID from the same stream as normal traffic.

        Sharing the minter matters: if fraudulent transactions drew IDs from a
        separate sequence, the ID space itself would leak the label.
        """
        return self.eco.new_id("transaction", "txn")


@dataclass
class AttackResult:
    transactions: List[Transaction]
    record: AttackRecord
    cover_transactions: List[Transaction] = field(default_factory=list)
    """Genuinely legitimate rows a family needs the ecosystem to contain.

    Some attacks are only realistic if unrelated honest traffic also touches the
    entities they create. A shell storefront that receives *nothing but* ring
    purchases makes a train-fitted per-merchant fraud rate a near-perfect oracle
    (measured at AUC 0.9822), because the merchant's history is 100% fraud by
    construction. Real shell merchants carry organic sales — that is what makes
    them plausible to the acquirer in the first place.

    These rows are ``is_fraud=False`` with ``attack_id=None`` and the pipeline
    folds them into the *normal* pool, so they neither consume the fraud budget
    nor appear in the ground-truth record.
    """


class EntityCollector:
    """Accumulates affected entity IDs for the ground-truth record.

    Deduplicates while preserving first-seen order, so records are stable across
    runs (a set would serialise in arbitrary order and break byte-identical
    reproducibility).
    """

    def __init__(self) -> None:
        self._buckets: Dict[str, List[str]] = {
            "customer": [], "merchant": [], "account": [],
            "device": [], "ip": [], "beneficiary": [],
        }
        self._seen: Dict[str, set] = {k: set() for k in self._buckets}

    def add(self, kind: str, entity_id: Optional[str]) -> None:
        if entity_id is None:
            return
        if entity_id not in self._seen[kind]:
            self._seen[kind].add(entity_id)
            self._buckets[kind].append(entity_id)

    def add_many(self, kind: str, entity_ids: Sequence[str]) -> None:
        for entity_id in entity_ids:
            self.add(kind, entity_id)

    def add_transaction_entities(self, txn: Transaction) -> None:
        """Register every entity a transaction touches."""
        self.add("account", txn.account_id)
        self.add("customer", txn.customer_id)
        self.add("merchant", txn.merchant_id)
        self.add("beneficiary", txn.beneficiary_id)
        self.add("device", txn.device_id)
        self.add("ip", txn.ip_id)

    def get(self, kind: str) -> List[str]:
        return list(self._buckets[kind])

    def as_record_kwargs(self) -> Dict[str, List[str]]:
        return {
            "affected_customer_ids": self.get("customer"),
            "affected_merchant_ids": self.get("merchant"),
            "affected_account_ids": self.get("account"),
            "affected_device_ids": self.get("device"),
            "affected_ip_ids": self.get("ip"),
            "affected_beneficiary_ids": self.get("beneficiary"),
        }


def attack_window(ctx: AttackContext, duration_hours: float) -> TimeWindow:
    """Choose a start time so the whole attack fits inside the split window.

    Containment is required: an attack straddling a split boundary would have
    part of its transactions in one split and part in another, corrupting the
    per-split prevalence accounting and the generalization design.
    """
    span = (ctx.window.end - ctx.window.start).total_seconds()
    duration_s = min(max(duration_hours, 0.25) * 3600.0, span * 0.9)
    latest_offset = max(0.0, span - duration_s)
    offset = float(ctx.rng.random()) * latest_offset
    start = ctx.window.start + timedelta(seconds=offset)
    return TimeWindow(start=start, end=start + timedelta(seconds=duration_s))


def spread_timestamps(
    ctx: AttackContext, window: TimeWindow, count: int, *, jitter: float = 0.35
) -> List[datetime]:
    """Distribute ``count`` timestamps across ``window``.

    Deliberately irregular. Perfectly even spacing would be a giveaway artifact
    (``docs/ML_READINESS.md`` §3) — real coordinated activity is bursty, and an
    exact cadence would let a detector cheat on inter-arrival regularity alone.
    """
    if count <= 0:
        return []
    span = (window.end - window.start).total_seconds()
    base = np.linspace(0.0, span, num=count, endpoint=False)
    step = span / max(1, count)
    noise = ctx.rng.normal(0.0, step * jitter, size=count)
    offsets = np.clip(base + noise, 0.0, max(0.0, span - 1.0))
    return sorted(window.start + timedelta(seconds=float(o)) for o in offsets)


def pick_accounts(
    ctx: AttackContext,
    count: int,
    *,
    min_activity: int = 0,
    exclude: Optional[Sequence[str]] = None,
) -> List[Account]:
    """Recruit existing legitimate accounts into an attack.

    Recruiting real accounts is what embeds a ring in the legitimate graph. The
    ``min_activity`` filter supports families that need an established victim
    (an account takeover is meaningless on an account with no history).

    Candidates must already exist when the attack window opens. Without that
    filter an attack could recruit an account created *later* in the same split —
    including a synthetic identity fabricated by an earlier cluster — and then
    transact on it before it existed. The audit measured ``account_age_days`` at
    -108 days on that path. A negative age is future information in its crudest
    form: a row that could not have been observed when it claims to have occurred.
    """
    excluded = set(exclude or ()) | set(ctx.exclude_accounts)
    start = ctx.window.start
    candidates = [
        a
        for a in ctx.eco.accounts
        if a.status == "active"
        and a.account_id not in excluded
        and a.created_at <= start
        and ctx.account_activity.get(a.account_id, 0) >= min_activity
    ]
    if not candidates:
        candidates = [
            a
            for a in ctx.eco.accounts
            if a.account_id not in excluded and a.created_at <= start
        ]
    if not candidates:  # pragma: no cover - defensive
        return []
    if count >= len(candidates):
        return list(candidates)
    idx = ctx.rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in sorted(idx)]


def entity_created_at(
    ctx: AttackContext, *, not_after: datetime, history_days: float
) -> datetime:
    """Draw a creation timestamp for an attack-created entity.

    Uses the *same* law legitimate entities obey
    (:func:`simulator.entities._created_at`): with probability
    ``new_entity_fraction`` the entity is a joiner appearing inside the observation
    window, otherwise it predates the window by up to ``history_days``. The result
    is clamped so the entity exists no later than ``not_after``.

    Sharing the law is the point. Attack families previously invented their own
    narrow ranges — payout beneficiaries were created 1-15 days before the split,
    ring beneficiaries 1-60 days — while legitimate beneficiaries spanned 500 days
    of history. The audit measured the consequence directly: beneficiary age at
    transaction time had a median of 31.9 days on fraudulent rows against 273.5 days
    on clean ones, an AUC of 0.9234 against ``circular_money_flow``. Any
    attack-created entity whose age distribution differs from its legitimate
    counterpart's hands a detector the label for free.
    """
    cfg = ctx.eco.config
    rng = ctx.rng
    if float(rng.random()) < cfg.ecosystem.new_entity_fraction:
        span = (not_after - cfg.warmup_start).total_seconds()
        if span > 0.0:
            return cfg.warmup_start + timedelta(seconds=float(rng.random()) * span)
    created = cfg.warmup_start - timedelta(
        days=float(rng.random()) * history_days + 1.0
    )
    return min(created, not_after - timedelta(hours=1.0))


def sample_status(ctx: AttackContext) -> str:
    """Status distribution for attack traffic.

    Kept close to the legitimate distribution on purpose. If fraud were always
    ``completed`` while normal traffic sometimes failed, ``status`` alone would
    betray the label.
    """
    u = float(ctx.rng.random())
    return "completed" if u < 0.955 else ("pending" if u < 0.99 else "failed")


def sample_channel(ctx: AttackContext) -> str:
    return CHANNELS[int(ctx.rng.choice(len(CHANNELS), p=[0.36, 0.40, 0.16, 0.08]))]


def jittered_amount(
    ctx: AttackContext, mean: float, std: float, *, floor: float = 0.5
) -> float:
    """Draw a continuous, non-round amount.

    Uniformly round amounts on fraud rows would be a giveaway artifact, so every
    attack amount goes through a continuous distribution and 2-decimal rounding
    identical to legitimate traffic.
    """
    value = float(ctx.rng.normal(mean, max(0.01, std)))
    return round_amount(max(floor, value))


def account_devices_ips(
    ctx: AttackContext, account: Account, ts: datetime
) -> Tuple[str, str]:
    """The account's own legitimate device and IP, as of ``ts``.

    Attack traffic defaults to the victim's real infrastructure. Only families
    where a new device is part of the scenario (takeover, ring-controlled
    devices) deviate — otherwise every fraud row would carry an unseen device and
    "new device" alone would separate the classes.

    Restricted to entities that already existed at ``ts``. The ecosystem grows
    continuously, so an account's device list contains hardware it had not yet
    acquired at an arbitrary earlier moment; drawing from the full list produced
    rows with device ages down to -128.8 days
    (``transaction_before_device_creation``, see
    :func:`simulator.validation._check_temporal_causality`). When nothing qualifies
    the oldest available entity is used, which is the only causally admissible
    choice.
    """
    devices = [
        d for d in (ctx.eco.account_devices.get(account.account_id) or ())
        if (dev := ctx.eco.device_by_id.get(d)) is not None and dev.created_at <= ts
    ]
    ips = [
        i for i in (ctx.eco.account_ips.get(account.account_id) or ())
        if (ip := ctx.eco.ip_by_id.get(i)) is not None and ip.created_at <= ts
    ]
    if devices:
        device_id = devices[int(ctx.rng.integers(0, len(devices)))]
    else:
        device_id = min(ctx.eco.devices, key=lambda d: d.created_at).device_id
    if ips:
        ip_id = ips[int(ctx.rng.integers(0, len(ips)))]
    else:
        ip_id = min(ctx.eco.ips, key=lambda i: i.created_at).ip_id
    return device_id, ip_id


def account_home(ctx: AttackContext, account: Account) -> Tuple[float, float]:
    """A point near the account holder's home, matching normal-traffic spread."""
    customer = ctx.eco.customer_by_id[account.customer_id]
    return offset_coordinates(
        customer.home_lat,
        customer.home_lon,
        float(ctx.rng.random()) * 8.0,
        float(ctx.rng.random()) * 2 * np.pi,
    )


def emit(
    ctx: AttackContext,
    *,
    timestamp: datetime,
    account: Account,
    amount: float,
    device_id: str,
    ip_id: str,
    lat: float,
    lon: float,
    transaction_type: str,
    channel: Optional[str] = None,
    merchant_id: Optional[str] = None,
    beneficiary_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Transaction:
    """Build one fraudulent transaction with the labels set consistently.

    Every family goes through here, which is what guarantees ``is_fraud`` and
    ``attack_id`` are always populated together — a row labelled fraudulent with
    no attack ID would be unattributable at evaluation time.
    """
    return Transaction(
        transaction_id=ctx.new_transaction_id(),
        timestamp=timestamp,
        account_id=account.account_id,
        customer_id=account.customer_id,
        merchant_id=merchant_id,
        beneficiary_id=beneficiary_id,
        device_id=device_id,
        ip_id=ip_id,
        amount=amount,
        currency=ctx.currency,
        channel=channel or sample_channel(ctx),
        transaction_type=transaction_type,
        lat=lat,
        lon=lon,
        status=status or sample_status(ctx),
        is_fraud=True,
        attack_id=ctx.attack_id,
    )


def emit_cover(
    ctx: AttackContext,
    *,
    timestamp: datetime,
    account: Account,
    amount: float,
    device_id: str,
    ip_id: str,
    lat: float,
    lon: float,
    transaction_type: str,
    channel: Optional[str] = None,
    merchant_id: Optional[str] = None,
    beneficiary_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Transaction:
    """Build one *legitimate* row that happens to touch attack-created entities.

    The mirror image of :func:`emit`: identical construction, but ``is_fraud`` is
    ``False`` and ``attack_id`` is ``None``. Used for cover traffic — see
    :attr:`AttackResult.cover_transactions` for why a shell merchant with a
    100%-fraud history is a data defect rather than a realistic one.

    Kept as a separate function rather than an ``is_fraud`` flag on :func:`emit`
    so that no call site can accidentally mislabel a fraudulent row as clean: the
    two paths are impossible to confuse at the point of use.
    """
    return Transaction(
        transaction_id=ctx.new_transaction_id(),
        timestamp=timestamp,
        account_id=account.account_id,
        customer_id=account.customer_id,
        merchant_id=merchant_id,
        beneficiary_id=beneficiary_id,
        device_id=device_id,
        ip_id=ip_id,
        amount=amount,
        currency=ctx.currency,
        channel=channel or sample_channel(ctx),
        transaction_type=transaction_type,
        lat=lat,
        lon=lon,
        status=status or sample_status(ctx),
        is_fraud=False,
        attack_id=None,
    )


def merchant_sampler(ctx: AttackContext) -> Callable[[datetime], str]:
    """Build a timestamp-aware sampler over existing legitimate merchants.

    Three properties matter. It **excludes fake merchants**, so a ring's purchases
    are not accidentally attributed to another attack's infrastructure. It samples
    **proportional to popularity**, exactly as normal traffic does — if attack
    purchases were spread uniformly over merchants while legitimate purchases
    concentrated on popular ones, the mismatch in merchant popularity would itself
    separate the classes. And it only returns merchants that **already existed** at
    the requested timestamp.

    The causality constraint is enforced by *rejection sampling* rather than by
    renormalising a filtered weight vector, because rejection preserves the
    popularity distribution over the eligible set exactly — the same technique
    :func:`simulator.transactions._choose_merchant` uses. After twelve failed draws
    it falls back to merchants that predate the warm-up boundary, which exist at
    every timestamp the dataset can contain.

    Returns a closure so the weight array is built once per attack, not per row.
    """
    ids = [m.merchant_id for m in ctx.eco.merchants if not m.is_fake]
    if not ids:  # pragma: no cover - defensive
        ids = [m.merchant_id for m in ctx.eco.merchants]
    by_id = ctx.eco.merchant_by_id
    weights = np.array([max(1e-9, ctx.eco.merchant_weight.get(m, 1.0)) for m in ids])
    weights = weights / weights.sum()
    created = [by_id[m].created_at for m in ids]

    floor = ctx.eco.config.warmup_start
    old_idx = [k for k, c in enumerate(created) if c <= floor]
    if old_idx:
        old_ids = [ids[k] for k in old_idx]
        old_weights = weights[old_idx] / weights[old_idx].sum()
    else:  # pragma: no cover - defensive
        oldest = int(np.argmin([c.timestamp() for c in created]))
        old_ids = [ids[oldest]]
        old_weights = np.array([1.0])

    def sample(ts: datetime) -> str:
        for _ in range(12):
            idx = int(ctx.rng.choice(len(ids), p=weights))
            if created[idx] <= ts:
                return ids[idx]
        return old_ids[int(ctx.rng.choice(len(old_ids), p=old_weights))]

    return sample


def vary_count(ctx: AttackContext, nominal: int, *, spread: float = 0.25, floor: int = 1) -> int:
    """Jitter a count parameter per attack instance, preserving its mean.

    Without this, every instance of a family inside one split has an *identical*
    footprint — measured before this helper existed: every ``fake_merchant_ring``
    in TRAIN produced exactly the same number of transactions. That is a giveaway
    artifact of the same kind as perfectly even timestamp spacing: a detector could
    learn "rings are exactly N rows" and score well for reasons that would never
    transfer to real data (``docs/ML_READINESS.md`` §3).

    The parameter still controls the distribution's centre, so parameter
    sensitivity (PRD FR-2) is unaffected — changing ``nominal`` still moves every
    realisation. ``spread`` is the relative standard deviation.
    """
    if nominal <= floor:
        return max(floor, nominal)
    jittered = float(ctx.rng.normal(nominal, max(0.0, spread) * nominal))
    return max(floor, int(round(jittered)))


def capped_entities(
    ctx: AttackContext,
    *,
    requested: int,
    existing: int,
    population: int,
    max_share: float,
) -> int:
    """How many *new* entities this attack may create before the cap binds.

    Families that fabricate entities — fake merchants, synthetic identities, payout
    beneficiaries — run once per attack instance, and the number of instances scales
    with the configured prevalence. Left unbounded, a 25% configuration drove fake
    merchants to ~72% of the merchant population, which turns "is this merchant
    fake?" into a question about the *base rate* rather than about behaviour, and
    undermines popularity-weighted sampling that assumes a mostly-legitimate
    population.

    Creation is therefore capped at ``max_share`` of the legitimate population.

    **May return 0.** That is the point: callers must top up to their requested ring
    size by *reusing* entities the ecosystem already has. A floor of 1 here would
    defeat the cap entirely, since hundreds of instances creating "just one" each is
    exactly the flood being prevented. Reuse is also the more realistic behaviour —
    a mature ring recycles shell infrastructure rather than endlessly onboarding new
    merchants — and it has the useful side effect of linking rings to each other,
    which makes ring-level attribution harder than isolated components would.
    """
    budget = int(population * max_share) - existing
    return max(0, min(requested, budget))


def severity_for(scale: float) -> str:
    """Map a 0..1 relative impact scale onto a severity label."""
    if scale >= 0.75:
        return "critical"
    if scale >= 0.5:
        return "high"
    if scale >= 0.25:
        return "medium"
    return "low"


def build_record(
    ctx: AttackContext,
    *,
    family: str,
    window: TimeWindow,
    severity: str,
    params: Dict[str, object],
    transactions: Sequence[Transaction],
    collector: EntityCollector,
) -> AttackRecord:
    """Assemble the ground-truth record for a completed attack."""
    if transactions:
        start = min(t.timestamp for t in transactions)
        end = max(t.timestamp for t in transactions)
    else:  # pragma: no cover - defensive
        start, end = window.start, window.end
    return AttackRecord(
        attack_id=ctx.attack_id,
        attack_type=family,
        start_time=start,
        end_time=end,
        severity=severity,
        generation_parameters=params,
        split=ctx.split,
        affected_transaction_ids=[t.transaction_id for t in transactions],
        **collector.as_record_kwargs(),
    )


#: Signature every family's ``generate`` function satisfies.
AttackGenerator = Callable[[AttackContext, object], AttackResult]
