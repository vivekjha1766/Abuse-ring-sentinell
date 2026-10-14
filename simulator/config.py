"""Typed, frozen configuration for the synthetic simulator.

Every knob that shapes the dataset lives here. No generator hardcodes a
magnitude, count, or attack shape: the configuration is the single source of
truth, it is captured verbatim in the dataset manifest, and identical config
plus identical seed reproduces an identical dataset.

Design rules
------------
* **Frozen dataclasses.** Configuration is immutable once built, so a generator
  cannot mutate shared state and silently break reproducibility.
* **No hardcoded attack shapes** (PRD FR-2). Each attack family has its own
  parameter dataclass; the generator reads it and nothing else.
* **Per-split parameter drift.** Attack parameters differ across TRAIN /
  VALIDATION / HELDOUT and generally intensify, per
  ``docs/DATA_SPLIT_PROTOCOL.md`` §6, so detectors are tested against attack
  *configurations* they have not seen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "AccountTakeoverParams",
    "AttackMix",
    "CircularMoneyFlowParams",
    "EcosystemConfig",
    "FakeMerchantRingParams",
    "MicrotransactionParams",
    "SharedDeviceRingParams",
    "SimulationConfig",
    "SplitConfig",
    "SyntheticIdentityParams",
    "TransactionSpikeParams",
    "SplitName",
    "SPLIT_NAMES",
    "default_config",
    "config_fingerprint",
]

SplitName = str
SPLIT_NAMES: Tuple[SplitName, ...] = ("train", "validation", "heldout")

#: Valid prevalence settings (``docs/ML_READINESS.md`` §4). 5% is the primary
#: synthetic benchmark configuration — not a real-world or industry rate.
SUPPORTED_PREVALENCES: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)


# --------------------------------------------------------------------------
# Attack parameter objects (PRD FR-2: each attack exposes a typed parameter object)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeMerchantRingParams:
    """Collusive merchants funnelling payments to shared beneficiaries."""

    n_merchants: int = 4
    n_accounts: int = 12
    n_devices: int = 3
    n_beneficiaries: int = 2
    duration_hours: float = 72.0
    intensity: float = 1.0
    """Multiplier on transaction volume within the ring."""
    amount_mean: float = 240.0
    amount_std: float = 90.0
    payout_ratio: float = 0.75
    """Fraction of collected value forwarded to ring beneficiaries."""
    max_fraud_share: float = 0.35
    """Ceiling on the share of a shell merchant's history that may be ring fraud.

    A shell earns its clean history organically — see
    :attr:`EcosystemConfig.shell_merchant_share` — so this is a backstop, not the
    primary mechanism. If a shell's organic volume is too thin to hold the ring
    rows below this share, the family tops it up with legitimate cover purchases
    across the shell's observable lifetime; when organic volume already suffices,
    nothing is injected and the dataset pays nothing.

    The measurements that make the backstop necessary. With no clean history at all,
    fake merchants held a 0.50-0.76 train fraud rate while legitimate merchants held
    0.008-0.118, only 1.5% of them above 0.27 — a single threshold on a train-fitted
    per-merchant fraud rate separated the ring at AUC 0.9830, and because 100% of
    held-out ring rows sat on shells already active in TRAIN the encoding
    transferred perfectly. With sizing by a fixed 0.55x-ring-rows ratio the rate
    stayed far too high; with an unconditional 0.10 target share the shortcut closed
    but cover traffic reached 27% of all rows at 25% prevalence.

    Lowering this value forces more top-up cover and costs dataset size; raising it
    makes the ring easier to find by its storefronts' reputations alone.
    """



@dataclass(frozen=True)
class SharedDeviceRingParams:
    """Many accounts operating from a small pool of device fingerprints."""

    n_accounts: int = 14
    n_devices: int = 3
    sharing_density: float = 0.7
    """Probability an account uses a ring device for a given transaction."""
    duration_hours: float = 96.0
    tx_per_account: int = 9
    amount_mean: float = 180.0
    amount_std: float = 70.0


@dataclass(frozen=True)
class TransactionSpikeParams:
    """A sudden burst of activity on one account."""

    multiplier: float = 5.0
    """Volume multiplier relative to the account's own baseline."""
    duration_hours: float = 6.0
    amount_scale: float = 1.4
    min_transactions: int = 8


@dataclass(frozen=True)
class CircularMoneyFlowParams:
    """Funds moving A -> B -> C -> A, returning near their origin."""

    cycle_length: int = 3
    amount: float = 1500.0
    rounds: int = 3
    duration_hours: float = 48.0
    decay: float = 0.93
    """Value retained per hop, modelling skimming."""


@dataclass(frozen=True)
class SyntheticIdentityParams:
    """A cluster of fabricated identities created close together in time."""

    n_identities: int = 8
    creation_window_hours: float = 48.0
    attribute_correlation: float = 0.8
    """Higher values make cluster members share more attributes."""
    ramp_days: float = 10.0
    tx_per_identity: int = 6
    amount_mean: float = 210.0
    incubation_days: Tuple[float, float] = (14.0, 420.0)
    """How long before the bust-out the batch of identities was fabricated.

    Synthetic identity fraud incubates: the identity is created, nursed with
    ordinary-looking activity until it looks established, and only then busted out.
    Creating the batch *inside* the attack window instead made every fabricated
    identity a few days old at the moment it transacted — measured at
    ``customer_age_days`` AUC 0.9970 with 99.6% recall, against only 1.04% of
    legitimate rows in the same age band. Entity age was therefore a near-perfect
    label rather than a behavioural signal.

    Incubation moves the *creation* earlier while the fraudulent bust-out stays
    fully inside its split window, so split containment is untouched. The family's
    signature remains the cluster structure: a batch created within
    ``creation_window_hours`` of one another, sharing attributes and infrastructure,
    busting out together.
    """
    history_activity_scale: float = 0.35
    """Incubation-period activity as a fraction of the identity's baseline rate.

    These history-building rows are legitimate — individually ordinary purchases,
    labelled ``is_fraud=False`` — so that "fraud" never degenerates into "any row
    touching a fabricated identity", which is precisely the quarantined
    ``customer.is_synthetic_identity`` oracle. Deliberately below 1.0: an identity
    under construction is nursed cheaply, not run at full tilt.
    """


@dataclass(frozen=True)
class MicrotransactionParams:
    """Many tiny payments that only look wrong in aggregate."""

    n_accounts: int = 10
    tx_per_account: int = 40
    amount_min: float = 0.6
    amount_max: float = 4.5
    duration_hours: float = 24.0


@dataclass(frozen=True)
class AccountTakeoverParams:
    """A legitimate account seized: new device, new IP, geographic jump."""

    new_device: bool = True
    new_ip: bool = True
    geo_jump_km: float = 1800.0
    velocity_multiplier: float = 6.0
    duration_hours: float = 8.0
    amount_scale: float = 2.2
    min_transactions: int = 6
    device_age_days: Tuple[float, float] = (0.05, 300.0)
    """How old the attacker's device is when the takeover starts, sampled
    log-uniformly. Originally this was a hardcoded ``U(0, 12)`` *hours*, which made
    ``device_age_days_at_txn`` an oracle (AUC 0.9844): every takeover device was
    freshly minted while every legitimate device predated the dataset. Attackers in
    practice reuse infrastructure they have held for months, so a broad range is
    both more faithful and no longer a giveaway. Parameterized rather than
    hardcoded, per PRD FR-2."""
    ip_age_days: Tuple[float, float] = (0.02, 220.0)
    """As above, for the attacker's IP (AUC 0.9855 before this change). Shorter
    ceiling than devices because addresses churn faster."""
    datacenter_ip_prob: float = 0.5
    """Probability the takeover IP is a datacenter/VPN address rather than an
    ordinary residential or mobile one."""
    ip_origin_weights: Tuple[float, float, float, float] = (0.30, 0.22, 0.28, 0.20)
    """Where the takeover connects *from*, as weights over four origins:
    ``(victim_own, shared_infrastructure, other_existing, new_dedicated)``.

    Minting a fresh address for every takeover made ``ip_account_degree`` an
    oracle. Because each new IP served exactly one victim and nothing else, an IP
    with a train-fitted account degree of 1 *was* a takeover: measured at 25%
    prevalence, 69.8% of takeover rows sat on a degree-1 IP and 29.8% on an IP
    unseen in TRAIN, with essentially nothing in between, against 2.4% of
    legitimate rows at degree 1. 311 of the 402 degree-1 addresses were
    takeover-touched. AUC reached 0.9820 at 25% and 0.9847 at 1%.

    The defect was the *uniqueness*, not the novelty. Real takeovers arrive over
    infrastructure that is usually already carrying somebody else's traffic:

    ``victim_own``
        A session hijacked behind the victim's own address — household wifi, their
        own CGNAT block. The address is entirely ordinary.
    ``shared_infrastructure``
        A NAT/CGNAT or corporate egress point already serving many unrelated
        accounts. Degree is *high*, not low.
    ``other_existing``
        An address already used by some other account: a residential proxy, a
        recycled VPN exit, a compromised home router. Degree looks legitimate
        because it is drawn from the legitimate population.
    ``new_dedicated``
        A freshly provisioned dedicated address. Retained as a real and common
        pattern — just no longer the *only* pattern, so degree 1 stops being a
        label.

    Deliberately leaves the takeover fully detectable by the signals that define
    the family: the geographic jump, the velocity burst, the amount scaling and the
    new device are all untouched. Only the address's sharing structure changes.
    Weights are normalised at use, so they need not sum to 1. Setting
    ``new_ip=False`` still forces ``victim_own`` outright."""


@dataclass(frozen=True)
class AttackMix:
    """How many instances of each family to inject into one split.

    Counts are *targets*. The pipeline honours the configured fraud prevalence
    by scaling instances, so realised counts may differ; both the target and the
    realised count are recorded in the manifest.
    """

    fake_merchant_ring: int = 2
    shared_device_ring: int = 2
    transaction_spike: int = 4
    circular_money_flow: int = 2
    synthetic_identity_cluster: int = 1
    microtransaction_attack: int = 2
    account_takeover: int = 3

    def as_mapping(self) -> Dict[str, int]:
        return dict(asdict(self))

    def total(self) -> int:
        return sum(self.as_mapping().values())


@dataclass(frozen=True)
class SplitConfig:
    """Attack configuration for a single temporal split.

    Parameters differ per split by design (``docs/DATA_SPLIT_PROTOCOL.md`` §6):
    ring sizes grow, cycles lengthen, spikes intensify. This tests
    generalization to unseen attack *configurations* — not to unseen attack
    *types*, a limitation stated explicitly in ``PRD.md`` §11.
    """

    name: SplitName
    mix: AttackMix = field(default_factory=AttackMix)
    fake_merchant_ring: FakeMerchantRingParams = field(default_factory=FakeMerchantRingParams)
    shared_device_ring: SharedDeviceRingParams = field(default_factory=SharedDeviceRingParams)
    transaction_spike: TransactionSpikeParams = field(default_factory=TransactionSpikeParams)
    circular_money_flow: CircularMoneyFlowParams = field(default_factory=CircularMoneyFlowParams)
    synthetic_identity_cluster: SyntheticIdentityParams = field(
        default_factory=SyntheticIdentityParams
    )
    microtransaction_attack: MicrotransactionParams = field(default_factory=MicrotransactionParams)
    account_takeover: AccountTakeoverParams = field(default_factory=AccountTakeoverParams)

    def params_for(self, family: str) -> Any:
        try:
            return getattr(self, family)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown attack family: {family!r}") from exc


@dataclass(frozen=True)
class EcosystemConfig:
    """Size and behaviour of the legitimate population.

    Defaults are portfolio-scale: large enough for meaningful graph structure,
    small enough to regenerate quickly.
    """

    n_customers: int = 900
    n_merchants: int = 140
    n_devices: int = 1000
    n_ips: int = 850
    n_beneficiaries: int = 1500
    """Payees available to legitimate accounts.

    Sized to be *mostly private*. A payee list is the most concentrated
    relationship in retail banking — people pay their own savings account, a
    landlord, a handful of friends — so most beneficiaries should be reachable from
    one or two accounts only. With the old pool of 260 shared uniformly across
    every transfer, each legitimate beneficiary was paid by dozens of unrelated
    accounts while an attack's payout beneficiary was paid by the handful of
    accounts in its ring. The adversarial audit measured the consequence:
    ``beneficiary_account_degree`` reached AUC 0.9889 against ``fake_merchant_ring``
    and 0.9683 against ``any_fraud`` (F1 0.835), and ``account_beneficiary_degree``
    reached 0.9846 against ``synthetic_identity_cluster`` — three oracles created
    entirely by the legitimate population being unrealistically promiscuous.
    """
    beneficiaries_per_account: Tuple[int, int] = (1, 4)
    """How many payees an account transacts with, drawn per account.

    Keeping this small is what puts legitimate payee degree in the same range as a
    ring's, so concentration cannot separate them on its own.
    """
    popular_beneficiary_fraction: float = 0.06
    """Fraction of beneficiaries that are *widely* paid — utilities, subscriptions,
    tax authorities. Their existence is what stops "many accounts pay this payee"
    from being suspicious by itself, mirroring the role shared NAT IPs play for
    ``shared_ip_pool_fraction``."""
    popular_beneficiary_prob: float = 0.45
    """Probability that a given payee slot is filled from the popular pool rather
    than with a private payee."""

    accounts_per_customer: Tuple[int, int] = (1, 3)

    shell_merchant_share: float = 0.35
    """Shell merchants pre-created as a fraction of ``n_merchants``.

    These carry ``is_fake=True`` and are the pool ``fake_merchant_ring`` recruits
    from. They are built *with the legitimate population*, before normal traffic
    runs, which is the entire point: the normal-traffic merchant sampler weights by
    popularity over every merchant that exists, so a shell accumulates organic
    legitimate purchases for free, across the whole timeline including warm-up.

    That replaces an earlier design in which the ring injected its own cover
    traffic after the fact. Injected cover worked as a shortcut fix but cost the
    dataset dearly: at 25% prevalence it added 75,555 rows to a 278,315-row
    dataset — 27% of every row was cover — which pushed realised prevalence down
    to 0.1594 against a configured 0.25, and because the cover spanned each
    shell's whole lifetime it landed mostly in the earliest split, skewing the
    split fractions to 0.683/0.167/0.150 against a configured 0.600/0.200/0.200.
    Organic traffic is drawn from the *normal* budget instead of added on top, so
    it costs neither prevalence nor split balance.

    Kept at or above :data:`simulator.attacks.fake_merchant_ring.
    MAX_FAKE_MERCHANT_SHARE` deliberately. If the pool were smaller than the cap,
    a high-prevalence run would exhaust it and start onboarding shells mid-run,
    after the normal-traffic sampler had already been fixed — those shells would
    have a 100%-fraudulent history, which is exactly the defect this removes.

    Shells the rings never recruit stay dormant: legitimate-looking storefronts
    flagged ``is_fake`` with no fraud on them. That weakens ``merchant.is_fake``
    as a label, which is acceptable — it is quarantined from features anyway
    (``docs/FEATURE_LEAKAGE_POLICY.md``) and attack attribution comes from
    ``AttackRecord.affected_merchant_ids``, not from the flag.
    """

    # --- legitimate sharing (ML_READINESS §3: benign overlap MUST exist) ---
    household_fraction: float = 0.16
    """Fraction of customers placed into a device-sharing household."""
    household_size: Tuple[int, int] = (2, 4)
    shared_ip_pool_fraction: float = 0.10
    """Fraction of IPs designated NAT/CGNAT/corporate egress (many users)."""
    shared_ip_users: Tuple[int, int] = (4, 20)

    # --- continuous ecosystem growth (ML_READINESS §3) ---
    new_entity_fraction: float = 0.24
    """Fraction of each *legitimate* entity population created **inside** the
    observation window rather than before it.

    This exists because of a measured leak, not for realism alone. When every
    legitimate entity predated the window, the only entities that appeared
    mid-timeline were the ones attacks created, so "this merchant is 12 days old"
    and "this device has never been seen in TRAIN" were near-perfect fraud oracles
    — the adversarial audit measured ``merchant_age_days_at_txn`` at AUC 0.9815,
    ``customer_age_days`` at 0.9998, and ``ip_unseen_in_train`` at precision 0.956.

    Real payment ecosystems onboard continuously: customers sign up, merchants are
    approved, handsets are replaced, IP leases rotate. Modelling that growth makes
    newness *common and mostly benign*, which is the only honest way to stop it
    being a shortcut. Entity age stays informative — it should be — but it can no
    longer stand alone."""
    new_device_adoption_prob: float = 0.26
    """Probability an account adopts a newly created device partway through the
    timeline (a phone upgrade). The device is unusable before it exists and becomes
    the account's primary afterwards, so new devices carry real volume rather than
    appearing once and never again."""
    new_ip_rotation_prob: float = 0.38
    """Same mechanism for IPs, at a higher rate: residential leases and mobile
    carrier egress genuinely do rotate more often than handsets are replaced."""

    # --- behavioural heterogeneity ---
    legit_outlier_rate: float = 0.012
    """Rate of genuinely large legitimate transactions."""
    legit_outlier_scale: Tuple[float, float] = (6.0, 40.0)
    micro_spend_rate: float = 0.07
    """Rate of genuinely *small* legitimate transactions, drawn independently of
    the account's personal mean — a premium account still buys coffee.

    Together with the ``micro`` spending profile
    (:data:`simulator.entities.SPENDING_PROFILES`) this populates the low end of
    the amount distribution with legitimate traffic. Without it the only tiny
    payments in the dataset are ``microtransaction_attack`` rows, which makes a
    single amount threshold a near-perfect detector for that family
    (``docs/ML_READINESS.md`` §3)."""
    micro_spend_range: Tuple[float, float] = (0.25, 9.0)
    """Sampled log-uniformly, so mass spreads across the whole low decade rather
    than piling up near the upper bound."""
    device_switch_prob: float = 0.10
    ip_switch_prob: float = 0.16

    # --- geography: legitimate mobility (ML_READINESS §3) ---
    travel_prob: float = 0.02
    """Probability of a one-off legitimate transaction away from home, *on top of*
    the coherent trips below. Reduced from its original value because trips now
    carry most legitimate mobility; this remains for genuinely isolated events."""
    trips_per_year: float = 3.2
    """Expected coherent trips per customer per year. A trip puts *all* of that
    customer's transactions near one destination for its whole duration, which
    matters: isolated far-away transactions left "several consecutive distant
    transactions" as a giveaway, because only a takeover produced a *run* of them."""
    trip_duration_days: Tuple[float, float] = (1.0, 16.0)
    trip_distance_range: Tuple[float, float] = (60.0, 1200.0)
    long_haul_share: float = 0.35
    """Share of trips that are long-haul rather than regional."""
    long_haul_range: Tuple[float, float] = (1200.0, 6800.0)
    """Long-haul trips must reach past the takeover ``geo_jump_km`` settings
    (1400 / 2100 / 3200 km). Before they did, ``distance_from_home_km`` was a
    *perfect* oracle for ``account_takeover`` — the audit measured AUC 1.0000,
    because no legitimate transaction was ever more than 900 km from home."""
    merchant_geolocated_share: float = 0.22
    """Share of card-not-present purchases geolocated to the **merchant** rather
    than the cardholder. This is how online payments actually resolve, and it is
    what turns the distance distribution from a tight spike at 0-8 km into a
    continuous heavy tail, so no distance threshold cleanly separates anything."""
    transfer_fraction: float = 0.18
    """Fraction of legitimate transactions that are transfers, not purchases."""
    large_transfer_prob: float = 0.13
    """Share of legitimate transfers that are genuinely large — rent, tuition,
    invoices, moving money between one's own accounts. Without these the only
    four-figure transfers in the dataset were ``circular_money_flow`` rows, and
    ``amount`` alone reached AUC 0.9783 on that family."""
    large_transfer_range: Tuple[float, float] = (600.0, 22000.0)
    """Sampled log-uniformly, so the whole range is populated rather than the top."""

    # --- seasonality (FR-1) ---
    hour_weights: Tuple[float, ...] = (
        0.25, 0.15, 0.10, 0.08, 0.08, 0.12,  # 00-05
        0.30, 0.65, 1.00, 1.20, 1.25, 1.30,  # 06-11
        1.35, 1.25, 1.15, 1.10, 1.15, 1.30,  # 12-17
        1.45, 1.50, 1.35, 1.05, 0.70, 0.40,  # 18-23
    )
    dow_weights: Tuple[float, ...] = (1.00, 0.98, 1.02, 1.05, 1.20, 1.30, 0.85)
    """Monday..Sunday."""


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration: the complete description of one dataset."""

    seed: int = 42
    fraud_prevalence: float = 0.05
    start: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    days: int = 120
    warmup_days: int = 14
    """Normal-only history before the evaluation window, so 7-day features have
    full history at the start of TRAIN (``docs/DATA_SPLIT_PROTOCOL.md`` §5)."""
    ecosystem: EcosystemConfig = field(default_factory=EcosystemConfig)
    splits: Tuple[SplitConfig, ...] = ()
    split_fractions: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    prevalence_tolerance: float = 0.30
    """Relative tolerance on realised vs configured prevalence."""
    currency: str = "XTS"
    """ISO 4217 code reserved for testing — never a real circulating currency."""

    def __post_init__(self) -> None:
        if self.fraud_prevalence <= 0 or self.fraud_prevalence >= 1:
            raise ValueError(f"fraud_prevalence must be in (0, 1), got {self.fraud_prevalence}")
        if self.days <= 0:
            raise ValueError("days must be positive")
        if self.warmup_days < 0:
            raise ValueError("warmup_days must be non-negative")
        if abs(sum(self.split_fractions) - 1.0) > 1e-9:
            raise ValueError(f"split_fractions must sum to 1.0, got {self.split_fractions}")
        if any(f <= 0 for f in self.split_fractions):
            raise ValueError("every split fraction must be positive")
        if not self.splits:
            object.__setattr__(self, "splits", default_splits())
        names = tuple(s.name for s in self.splits)
        if names != SPLIT_NAMES:
            raise ValueError(f"splits must be exactly {SPLIT_NAMES}, got {names}")

    # -- derived time boundaries ------------------------------------------
    @property
    def warmup_start(self) -> datetime:
        return self.start - timedelta(days=self.warmup_days)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.days)

    def split_config(self, name: SplitName) -> SplitConfig:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(f"unknown split: {name!r}")

    def with_prevalence(self, prevalence: float) -> "SimulationConfig":
        """Return a copy at a different fraud prevalence (1/5/10/25%)."""
        return replace(self, fraud_prevalence=prevalence)

    def with_seed(self, seed: int) -> "SimulationConfig":
        return replace(self, seed=seed)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict; recorded verbatim in the dataset manifest."""
        return _jsonify(asdict(self))


def _jsonify(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def config_fingerprint(config: SimulationConfig) -> str:
    """Stable short hash of a configuration, used in the manifest.

    Two configs with the same fingerprint and the same seed must produce the
    same dataset — that property is asserted by the determinism tests.
    """
    import hashlib

    blob = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Default per-split attack configurations
# --------------------------------------------------------------------------


def default_splits() -> Tuple[SplitConfig, ...]:
    """Attack parameters per split, intensifying toward the held-out period.

    The progression implements the table in ``docs/DATA_SPLIT_PROTOCOL.md`` §6:
    ring size small -> medium -> larger, cycle length 3 -> 4/5 -> 6, spike
    multiplier ~5x -> ~7x -> ~10x, parameter distribution narrower -> shifted ->
    broader.

    **Why the per-instance footprints are deliberately modest.** Every split must
    contain at least one instance of all seven families
    (``docs/ML_READINESS.md`` §5), so the sum of one instance of each family is a
    *floor* on that split's fraud row count — and therefore a floor on achievable
    prevalence. With ~30k normal transactions in the smaller splits, a 1% target
    allows only ~300 fraud rows, so the seven-family floor has to fit inside that.
    Footprints here sum to roughly 145 / 235 / 315 rows for train / validation /
    heldout, which keeps the 1% configuration reachable
    (see :func:`simulator.pipeline.fraud_budget`).

    The escalation across splits is therefore expressed in attack *character* —
    density, correlation, cycle length, velocity, geographic reach — rather than in
    raw transaction volume. Making heldout attacks merely *bigger* would be a weak
    form of distribution shift anyway: a detector could ride volume alone.
    """
    train = SplitConfig(
        name="train",
        mix=AttackMix(
            fake_merchant_ring=2,
            shared_device_ring=2,
            transaction_spike=5,
            circular_money_flow=2,
            synthetic_identity_cluster=1,
            microtransaction_attack=2,
            account_takeover=4,
        ),
        fake_merchant_ring=FakeMerchantRingParams(
            n_merchants=3, n_accounts=6, n_devices=2, n_beneficiaries=1,
            duration_hours=72.0, intensity=1.6, amount_mean=210.0, amount_std=80.0,
        ),
        shared_device_ring=SharedDeviceRingParams(
            n_accounts=6, n_devices=2, sharing_density=0.65,
            duration_hours=96.0, tx_per_account=4, amount_mean=165.0, amount_std=60.0,
        ),
        transaction_spike=TransactionSpikeParams(
            multiplier=5.0, duration_hours=6.0, amount_scale=1.3, min_transactions=8
        ),
        circular_money_flow=CircularMoneyFlowParams(
            cycle_length=3, amount=1200.0, rounds=3, duration_hours=48.0, decay=0.94
        ),
        synthetic_identity_cluster=SyntheticIdentityParams(
            n_identities=4, creation_window_hours=60.0, attribute_correlation=0.75,
            ramp_days=10.0, tx_per_identity=3, amount_mean=190.0,
        ),
        microtransaction_attack=MicrotransactionParams(
            n_accounts=4, tx_per_account=12, amount_min=0.7, amount_max=4.0,
            duration_hours=24.0,
        ),
        account_takeover=AccountTakeoverParams(
            geo_jump_km=1400.0, velocity_multiplier=5.0, duration_hours=8.0,
            amount_scale=2.0, min_transactions=6,
        ),
    )

    validation = SplitConfig(
        name="validation",
        mix=AttackMix(
            fake_merchant_ring=2,
            shared_device_ring=2,
            transaction_spike=4,
            circular_money_flow=2,
            synthetic_identity_cluster=1,
            microtransaction_attack=2,
            account_takeover=3,
        ),
        fake_merchant_ring=FakeMerchantRingParams(
            n_merchants=4, n_accounts=7, n_devices=3, n_beneficiaries=2,
            duration_hours=60.0, intensity=1.7, amount_mean=260.0, amount_std=105.0,
        ),
        shared_device_ring=SharedDeviceRingParams(
            n_accounts=8, n_devices=3, sharing_density=0.72,
            duration_hours=84.0, tx_per_account=5, amount_mean=195.0, amount_std=80.0,
        ),
        transaction_spike=TransactionSpikeParams(
            multiplier=7.0, duration_hours=5.0, amount_scale=1.5, min_transactions=10
        ),
        circular_money_flow=CircularMoneyFlowParams(
            cycle_length=4, amount=1800.0, rounds=4, duration_hours=36.0, decay=0.92
        ),
        synthetic_identity_cluster=SyntheticIdentityParams(
            n_identities=6, creation_window_hours=42.0, attribute_correlation=0.82,
            ramp_days=8.0, tx_per_identity=4, amount_mean=225.0,
        ),
        microtransaction_attack=MicrotransactionParams(
            n_accounts=5, tx_per_account=16, amount_min=0.5, amount_max=5.0,
            duration_hours=18.0,
        ),
        account_takeover=AccountTakeoverParams(
            geo_jump_km=2100.0, velocity_multiplier=7.0, duration_hours=6.0,
            amount_scale=2.4, min_transactions=7,
        ),
    )

    heldout = SplitConfig(
        name="heldout",
        mix=AttackMix(
            fake_merchant_ring=2,
            shared_device_ring=2,
            transaction_spike=4,
            circular_money_flow=2,
            synthetic_identity_cluster=1,
            microtransaction_attack=2,
            account_takeover=3,
        ),
        fake_merchant_ring=FakeMerchantRingParams(
            n_merchants=5, n_accounts=7, n_devices=4, n_beneficiaries=3,
            duration_hours=48.0, intensity=1.8, amount_mean=310.0, amount_std=140.0,
        ),
        shared_device_ring=SharedDeviceRingParams(
            n_accounts=10, n_devices=4, sharing_density=0.8,
            duration_hours=72.0, tx_per_account=5, amount_mean=230.0, amount_std=100.0,
        ),
        transaction_spike=TransactionSpikeParams(
            multiplier=10.0, duration_hours=4.0, amount_scale=1.8, min_transactions=12
        ),
        circular_money_flow=CircularMoneyFlowParams(
            cycle_length=6, amount=2400.0, rounds=5, duration_hours=30.0, decay=0.90
        ),
        synthetic_identity_cluster=SyntheticIdentityParams(
            n_identities=8, creation_window_hours=30.0, attribute_correlation=0.88,
            ramp_days=6.0, tx_per_identity=5, amount_mean=270.0,
        ),
        microtransaction_attack=MicrotransactionParams(
            n_accounts=5, tx_per_account=20, amount_min=0.4, amount_max=6.0,
            duration_hours=12.0,
        ),
        account_takeover=AccountTakeoverParams(
            geo_jump_km=3200.0, velocity_multiplier=10.0, duration_hours=5.0,
            amount_scale=3.0, min_transactions=9,
        ),
    )

    return (train, validation, heldout)


def default_config(seed: int = 42, prevalence: float = 0.05) -> SimulationConfig:
    """The primary synthetic benchmark configuration.

    5% prevalence is a *benchmark choice*, not a real-world or industry fraud
    rate (``PRD.md`` §8).
    """
    if prevalence not in SUPPORTED_PREVALENCES:
        raise ValueError(
            f"prevalence must be one of {SUPPORTED_PREVALENCES}, got {prevalence}"
        )
    return SimulationConfig(seed=seed, fraud_prevalence=prevalence, splits=default_splits())
