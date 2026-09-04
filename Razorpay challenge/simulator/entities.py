"""Synthetic entity generation — the six node types of the ecosystem.

Field names and types match ``DATA_DICTIONARY.md`` §2 exactly, because the
Checkpoint 4 database schema and the Checkpoint 3 graph are both built from
these definitions.

Two properties here matter more than the rest, both required by
``docs/ML_READINESS.md`` §3:

**Benign sharing must exist.** Households share devices and NAT/CGNAT egress
points share IPs. Without that baseline, "shared device implies fraud" would be
trivially true and the entire graph benchmark would be meaningless. The
generator therefore builds legitimate sharing structure *first*, so rings are
later embedded into a population that already shares infrastructure.

**Heterogeneity, not uniformity.** Merchant volumes, customer spending
profiles, and account frequencies all vary widely, so no single threshold
separates the classes.

**The population grows over time.** A fixed cast of entities all predating the
data is the single most dangerous simplification available here: if only attacks
introduce new merchants, devices, IPs and customers, then entity *age* and
"unseen before now" become near-perfect fraud oracles. Legitimate entities are
therefore created continuously across the whole observation window — see
:func:`_created_at` and
:attr:`simulator.config.EcosystemConfig.new_entity_fraction`.

No real PII exists here. Geography uses invented country codes, and IP
addresses are drawn only from IANA special-purpose ranges that can never route
to a real host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator.base_generator import IdMinter, SeedRegistry, clamp, offset_coordinates
from simulator.config import EcosystemConfig, SimulationConfig

__all__ = [
    "Account",
    "Beneficiary",
    "Customer",
    "Device",
    "Ecosystem",
    "IPAddress",
    "Merchant",
    "SEGMENT_TX_FREQUENCY",
    "generate_ecosystem",
    "shuffle_entity_tables",
]

# --------------------------------------------------------------------------
# Synthetic reference data. Every value below is invented.
# --------------------------------------------------------------------------

#: Invented country codes with an arbitrary anchor coordinate. The "X*" prefix
#: range is user-assignable in ISO 3166-1, so these can never denote a real
#: country.
SYNTHETIC_COUNTRIES: Tuple[Tuple[str, float, float], ...] = (
    ("XA", 12.5, 77.6),
    ("XB", 48.9, 2.4),
    ("XC", -23.5, -46.6),
    ("XD", 35.7, 139.7),
    ("XE", 52.5, 13.4),
    ("XF", -33.9, 151.2),
    ("XG", 40.7, -74.0),
    ("XH", 1.35, 103.8),
)
REGIONS_PER_COUNTRY: Tuple[str, ...] = ("north", "south", "east", "west", "central")

MERCHANT_CATEGORIES: Tuple[str, ...] = (
    "retail", "grocery", "travel", "digital_goods", "restaurants", "fuel",
    "electronics", "healthcare", "entertainment", "services",
)
MERCHANT_SIZE_TIERS: Tuple[str, ...] = ("small", "medium", "large")
CUSTOMER_SEGMENTS: Tuple[str, ...] = ("low_activity", "medium_activity", "high_activity")
#: ``micro`` exists for a specific reason. Without a genuine low-value legitimate
#: segment, the only sub-currency-unit payments in the dataset are
#: ``microtransaction_attack`` rows, and the rule "amount < 6" becomes a near-perfect
#: detector for that family — measured at 100% recall and 79% precision before this
#: profile was added. Real payment ecosystems have a substantial small-ticket tier
#: (transit, coffee, in-app purchases, tips), so this is a realism fix, not a
#: difficulty knob: see ``docs/ML_READINESS.md`` §3.
SPENDING_PROFILES: Tuple[str, ...] = ("micro", "budget", "standard", "premium", "erratic")
RISK_BANDS: Tuple[str, ...] = ("A", "B", "C", "D")
ACCOUNT_TYPES: Tuple[str, ...] = ("checking", "savings", "wallet")
ACCOUNT_STATUSES: Tuple[str, ...] = ("active", "dormant", "closed")

#: Baseline transactions/day by customer segment. The single source of truth for
#: how active an account is, shared by the legitimate population and by
#: :mod:`simulator.attacks.synthetic_identity_cluster`.
#:
#: Fabricated identities previously drew from a private ``uniform(0.1, 0.6)`` and
#: emitted a fixed six transactions each, which left them far quieter than any real
#: account: median 4 lifetime transactions against 17 for a genuinely new
#: legitimate account. Because the audit scores *rows*, and quiet accounts
#: contribute few rows, that made ``account_transaction_count`` rank almost every
#: synthetic-identity row below almost every other row — measured at AUC 0.9961.
#: Synthetic identity fraud works by *building* a credible history before the
#: bust-out, so drawing volume from the same law as legitimate accounts is both the
#: fix and the more faithful behaviour. The family's signature is the cluster
#: structure, never "this account is unusually quiet".
SEGMENT_TX_FREQUENCY: Dict[str, Tuple[float, float]] = {
    "low_activity": (0.06, 0.35),
    "medium_activity": (0.35, 1.1),
    "high_activity": (1.1, 3.4),
}
DEVICE_TYPES: Tuple[str, ...] = ("mobile", "desktop", "tablet")
OS_FAMILIES: Tuple[str, ...] = ("os_alpha", "os_beta", "os_gamma", "os_delta")
IP_TYPES: Tuple[str, ...] = ("residential", "mobile", "datacenter", "vpn")
BENEFICIARY_TYPES: Tuple[str, ...] = ("individual", "business")

#: IANA special-purpose IPv4 blocks: TEST-NET-1/2/3, benchmarking, and the
#: shared-address (CGNAT) space. None of these are globally routable.
_RESERVED_BLOCKS: Tuple[Tuple[str, int, int], ...] = (
    ("192.0.2", 1, 255),      # TEST-NET-1 (RFC 5737)
    ("198.51.100", 1, 255),   # TEST-NET-2 (RFC 5737)
    ("203.0.113", 1, 255),    # TEST-NET-3 (RFC 5737)
    ("198.18.0", 1, 255),     # Benchmarking (RFC 2544)
    ("198.19.0", 1, 255),     # Benchmarking (RFC 2544)
)
_CGNAT_PREFIX = "100.64"      # Shared address space (RFC 6598)


# --------------------------------------------------------------------------
# Entity dataclasses (DATA_DICTIONARY.md §2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Customer:
    customer_id: str
    created_at: datetime
    country: str
    region: str
    home_lat: float
    home_lon: float
    segment: str
    spending_profile: str
    risk_band: str
    """Generator-side *behavioural* band. Explicitly NOT a fraud label: it is
    assigned from spending volatility alone and is independent of attacks."""
    is_synthetic_identity: bool = False
    """Ground-truth marker. Quarantined from features."""


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    created_at: datetime
    category: str
    country: str
    lat: float
    lon: float
    size_tier: str
    avg_ticket_size: float
    is_fake: bool = False
    """Ground-truth marker. Quarantined from features."""


@dataclass(frozen=True)
class Account:
    account_id: str
    customer_id: str
    created_at: datetime
    account_type: str
    status: str
    baseline_tx_frequency: float
    """Expected transactions per day."""
    baseline_amount_mean: float
    baseline_amount_std: float


@dataclass(frozen=True)
class Device:
    device_id: str
    created_at: datetime
    device_type: str
    os_family: str
    fingerprint_hash: str
    is_shared_legitimate: bool = False
    """True for household/shared machines — benign overlap, by design."""


@dataclass(frozen=True)
class IPAddress:
    ip_id: str
    created_at: datetime
    ip_address: str
    country: str
    region: str
    lat: float
    lon: float
    ip_type: str
    is_shared_legitimate: bool = False
    """True for NAT / CGNAT / corporate egress — benign overlap, by design."""


@dataclass(frozen=True)
class Beneficiary:
    beneficiary_id: str
    created_at: datetime
    beneficiary_type: str
    country: str
    linked_account_id: Optional[str] = None
    """Optional link back to an account, which is what makes cycles possible."""


# --------------------------------------------------------------------------
# Ecosystem container
# --------------------------------------------------------------------------


@dataclass
class Ecosystem:
    """All entities plus the relationship pools the generators need.

    Mutable on purpose: attack families add entities (fake merchants, synthetic
    identities, takeover devices) through the ``add_*`` methods, which keep ID
    minting centralised so attack-created entities are indistinguishable from
    ordinary ones by ID alone.
    """

    config: SimulationConfig
    minter: IdMinter

    customers: List[Customer] = field(default_factory=list)
    merchants: List[Merchant] = field(default_factory=list)
    accounts: List[Account] = field(default_factory=list)
    devices: List[Device] = field(default_factory=list)
    ips: List[IPAddress] = field(default_factory=list)
    beneficiaries: List[Beneficiary] = field(default_factory=list)

    #: account_id -> device_ids that account legitimately uses
    account_devices: Dict[str, List[str]] = field(default_factory=dict)
    #: account_id -> ip_ids that account legitimately uses
    account_ips: Dict[str, List[str]] = field(default_factory=dict)
    #: account_id -> beneficiary_ids that account legitimately pays. A payee list,
    #: not a global pool: see EcosystemConfig.beneficiaries_per_account for the
    #: oracles that a globally shared pool produced.
    account_beneficiaries: Dict[str, List[str]] = field(default_factory=dict)
    #: customer_id -> account_ids
    customer_accounts: Dict[str, List[str]] = field(default_factory=dict)
    #: account_id -> (device_id, adopted_at) for a device the account starts using
    #: partway through the timeline. Kept separate from ``account_devices`` so the
    #: generator can honour "you cannot use a phone you have not bought yet".
    account_device_adoption: Dict[str, Tuple[str, datetime]] = field(default_factory=dict)
    #: account_id -> (ip_id, rotated_at), same idea for address rotation.
    account_ip_adoption: Dict[str, Tuple[str, datetime]] = field(default_factory=dict)
    #: per-merchant relative popularity, driving heterogeneous volumes
    merchant_weight: Dict[str, float] = field(default_factory=dict)

    _counters: Dict[str, int] = field(default_factory=dict)
    _ip_cursor: int = 0

    # -- lookups ----------------------------------------------------------
    @property
    def customer_by_id(self) -> Dict[str, Customer]:
        return {c.customer_id: c for c in self.customers}

    @property
    def account_by_id(self) -> Dict[str, Account]:
        return {a.account_id: a for a in self.accounts}

    @property
    def merchant_by_id(self) -> Dict[str, Merchant]:
        return {m.merchant_id: m for m in self.merchants}

    @property
    def device_by_id(self) -> Dict[str, Device]:
        return {d.device_id: d for d in self.devices}

    @property
    def ip_by_id(self) -> Dict[str, IPAddress]:
        return {i.ip_id: i for i in self.ips}

    @property
    def beneficiary_by_id(self) -> Dict[str, Beneficiary]:
        return {b.beneficiary_id: b for b in self.beneficiaries}

    def entity_counts(self) -> Dict[str, int]:
        return {
            "customers": len(self.customers),
            "merchants": len(self.merchants),
            "accounts": len(self.accounts),
            "devices": len(self.devices),
            "ips": len(self.ips),
            "beneficiaries": len(self.beneficiaries),
        }

    # -- ID minting -------------------------------------------------------
    def _next_index(self, kind: str) -> int:
        idx = self._counters.get(kind, 0)
        self._counters[kind] = idx + 1
        return idx

    def new_id(self, kind: str, prefix: str) -> str:
        return self.minter.mint(prefix, self._next_index(kind))

    def next_ip_address(self) -> str:
        """Allocate the next reserved-range address, never repeating."""
        cursor = self._ip_cursor
        self._ip_cursor += 1
        total_reserved = sum(hi - lo for _, lo, hi in _RESERVED_BLOCKS)
        if cursor < total_reserved:
            for prefix, lo, hi in _RESERVED_BLOCKS:
                size = hi - lo
                if cursor < size:
                    return f"{prefix}.{lo + cursor}"
                cursor -= size
        # Overflow into CGNAT space, which is vast and equally unroutable.
        offset = cursor - total_reserved
        return f"{_CGNAT_PREFIX}.{(offset // 254) % 256}.{offset % 254 + 1}"

    # -- entity creation used by both base generation and attacks ---------
    def add_customer(self, **kwargs: object) -> Customer:
        customer = Customer(customer_id=self.new_id("customer", "cust"), **kwargs)  # type: ignore[arg-type]
        self.customers.append(customer)
        self.customer_accounts.setdefault(customer.customer_id, [])
        return customer

    def add_merchant(self, *, weight: float = 1.0, **kwargs: object) -> Merchant:
        merchant = Merchant(merchant_id=self.new_id("merchant", "mer"), **kwargs)  # type: ignore[arg-type]
        self.merchants.append(merchant)
        self.merchant_weight[merchant.merchant_id] = weight
        return merchant

    def add_account(self, **kwargs: object) -> Account:
        account = Account(account_id=self.new_id("account", "acc"), **kwargs)  # type: ignore[arg-type]
        self.accounts.append(account)
        self.customer_accounts.setdefault(account.customer_id, []).append(account.account_id)
        self.account_devices.setdefault(account.account_id, [])
        self.account_ips.setdefault(account.account_id, [])
        self.account_beneficiaries.setdefault(account.account_id, [])
        return account

    def add_device(self, **kwargs: object) -> Device:
        device = Device(device_id=self.new_id("device", "dev"), **kwargs)  # type: ignore[arg-type]
        self.devices.append(device)
        return device

    def add_ip(self, **kwargs: object) -> IPAddress:
        ip = IPAddress(ip_id=self.new_id("ip", "ip"), **kwargs)  # type: ignore[arg-type]
        self.ips.append(ip)
        return ip

    def add_beneficiary(self, **kwargs: object) -> Beneficiary:
        ben = Beneficiary(beneficiary_id=self.new_id("beneficiary", "ben"), **kwargs)  # type: ignore[arg-type]
        self.beneficiaries.append(ben)
        return ben


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _fingerprint(rng: np.random.Generator) -> str:
    return "fp_" + "".join(f"{int(b):02x}" for b in rng.integers(0, 256, size=8))


def _pick_location(
    rng: np.random.Generator, country_idx: int, spread_km: float = 60.0
) -> Tuple[str, str, float, float]:
    code, base_lat, base_lon = SYNTHETIC_COUNTRIES[country_idx]
    region = REGIONS_PER_COUNTRY[int(rng.integers(0, len(REGIONS_PER_COUNTRY)))]
    lat, lon = offset_coordinates(
        base_lat, base_lon, float(rng.random()) * spread_km, float(rng.random()) * 2 * np.pi
    )
    return code, region, lat, lon


def _created_at(
    rng: np.random.Generator,
    config: SimulationConfig,
    cfg: EcosystemConfig,
    *,
    history_days: float,
) -> datetime:
    """Draw a creation timestamp for a legitimate entity.

    With probability ``new_entity_fraction`` the entity is a *joiner*: it comes into
    existence at a uniformly random instant inside the observation window, so the
    dataset contains a continuous stream of new legitimate entities across TRAIN,
    VALIDATION and HELDOUT alike. Otherwise it predates the window by up to
    ``history_days``.

    This one function is the fix for five of the thirteen oracles the adversarial
    audit found. See :attr:`simulator.config.EcosystemConfig.new_entity_fraction`
    for the measurements.
    """
    if float(rng.random()) < cfg.new_entity_fraction:
        span = (config.end - config.warmup_start).total_seconds()
        return config.warmup_start + timedelta(seconds=float(rng.random()) * span)
    return config.warmup_start - timedelta(days=float(rng.random()) * history_days + 1.0)


def _generate_devices(eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig) -> None:
    rng = seeds.rng("entities/devices")
    for _ in range(cfg.n_devices):
        created = _created_at(rng, eco.config, cfg, history_days=240.0)
        eco.add_device(
            created_at=created,
            device_type=DEVICE_TYPES[int(rng.integers(0, len(DEVICE_TYPES)))],
            os_family=OS_FAMILIES[int(rng.integers(0, len(OS_FAMILIES)))],
            fingerprint_hash=_fingerprint(rng),
            is_shared_legitimate=False,
        )


def _generate_ips(eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig) -> None:
    rng = seeds.rng("entities/ips")
    n_shared = max(1, int(cfg.n_ips * cfg.shared_ip_pool_fraction))
    for i in range(cfg.n_ips):
        country_idx = int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
        code, region, lat, lon = _pick_location(rng, country_idx, spread_km=120.0)
        shared = i < n_shared
        # Shared egress points are typically datacenter/corporate NAT.
        ip_type = "datacenter" if shared and rng.random() < 0.5 else IP_TYPES[
            int(rng.integers(0, len(IP_TYPES)))
        ]
        eco.add_ip(
            created_at=_created_at(rng, eco.config, cfg, history_days=200.0),
            ip_address=eco.next_ip_address(),
            country=code,
            region=region,
            lat=lat,
            lon=lon,
            ip_type=ip_type,
            is_shared_legitimate=shared,
        )


def _generate_merchants(eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig) -> None:
    """Merchants with deliberately heterogeneous volume (PRD FR-1).

    Size tier drives both the popularity weight and the typical ticket size, so
    a few large merchants carry most traffic while many small ones carry little
    — a heavy-tailed shape rather than a uniform one.
    """
    rng = seeds.rng("entities/merchants")
    tier_probs = np.array([0.62, 0.28, 0.10])
    tier_weight = {"small": (0.4, 3.0), "medium": (3.0, 12.0), "large": (12.0, 60.0)}
    tier_ticket = {"small": (8.0, 45.0), "medium": (25.0, 140.0), "large": (60.0, 420.0)}
    for _ in range(cfg.n_merchants):
        tier = MERCHANT_SIZE_TIERS[int(rng.choice(len(MERCHANT_SIZE_TIERS), p=tier_probs))]
        w_lo, w_hi = tier_weight[tier]
        t_lo, t_hi = tier_ticket[tier]
        country_idx = int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
        code, _region, lat, lon = _pick_location(rng, country_idx, spread_km=90.0)
        eco.add_merchant(
            weight=float(rng.uniform(w_lo, w_hi)),
            created_at=_created_at(rng, eco.config, cfg, history_days=900.0),
            category=MERCHANT_CATEGORIES[int(rng.integers(0, len(MERCHANT_CATEGORIES)))],
            country=code,
            lat=lat,
            lon=lon,
            size_tier=tier,
            avg_ticket_size=round(float(rng.uniform(t_lo, t_hi)), 2),
            is_fake=False,
        )

    # --- shell merchant pool -------------------------------------------------
    # Storefronts that exist to be recruited by ``fake_merchant_ring``. They are
    # created *here*, with the legitimate population, rather than by the attack, so
    # that ``simulator.transactions._choose_merchant`` — which weights by popularity
    # over every merchant in existence and does not filter ``is_fake`` — gives each
    # one an organic legitimate trading history across the whole timeline. That
    # history is what stops a shell from having a 100%-fraudulent past, and unlike
    # injected cover traffic it comes out of the normal-traffic budget rather than
    # being added on top of it. See
    # :attr:`simulator.config.EcosystemConfig.shell_merchant_share`.
    #
    # A separate RNG stream keeps the legitimate merchants' realisation unchanged
    # when the pool size is tuned. Shells are drawn from the *small* tier alone: the
    # tier fixes ``avg_ticket_size`` as well as popularity, and ring purchase amounts
    # come from the family's own parameters, so a large-tier shell would carry a
    # ticket size its own traffic contradicts. Small-tier shells are also what the
    # audit has already measured as safe.
    shell_rng = seeds.rng("entities/shell_merchants")
    n_shells = int(round(cfg.n_merchants * max(0.0, cfg.shell_merchant_share)))
    w_lo, w_hi = tier_weight["small"]
    t_lo, t_hi = tier_ticket["small"]
    for _ in range(n_shells):
        country_idx = int(shell_rng.integers(0, len(SYNTHETIC_COUNTRIES)))
        code, _region, lat, lon = _pick_location(shell_rng, country_idx, spread_km=90.0)
        eco.add_merchant(
            weight=float(shell_rng.uniform(w_lo, w_hi)),
            created_at=_created_at(shell_rng, eco.config, cfg, history_days=900.0),
            category=MERCHANT_CATEGORIES[
                int(shell_rng.integers(0, len(MERCHANT_CATEGORIES)))
            ],
            country=code,
            lat=lat,
            lon=lon,
            size_tier="small",
            avg_ticket_size=round(float(shell_rng.uniform(t_lo, t_hi)), 2),
            is_fake=True,
        )


def _generate_customers_and_accounts(
    eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig
) -> None:
    rng = seeds.rng("entities/customers")
    seg_probs = np.array([0.45, 0.38, 0.17])
    profile_probs = np.array([0.11, 0.30, 0.36, 0.14, 0.09])
    seg_freq = SEGMENT_TX_FREQUENCY
    # ``micro`` accounts sit in the same amount range microtransaction attacks use,
    # so small size alone carries no signal; they are separated from the attack by
    # *aggregate* behaviour (how many, how fast, to whom), which is the point.
    profile_amount = {"micro": (1.1, 6.5), "budget": (12.0, 45.0),
                      "standard": (35.0, 120.0), "premium": (110.0, 420.0),
                      "erratic": (20.0, 260.0)}
    profile_cv = {"micro": 0.7, "budget": 0.45, "standard": 0.6, "premium": 0.75,
                  "erratic": 1.25}

    lo_acc, hi_acc = cfg.accounts_per_customer
    for _ in range(cfg.n_customers):
        segment = CUSTOMER_SEGMENTS[int(rng.choice(len(CUSTOMER_SEGMENTS), p=seg_probs))]
        profile = SPENDING_PROFILES[int(rng.choice(len(SPENDING_PROFILES), p=profile_probs))]
        country_idx = int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
        code, region, lat, lon = _pick_location(rng, country_idx)
        # risk_band from spending volatility only — never from fraud status.
        cv = profile_cv[profile]
        band_idx = int(clamp(cv * 2.2 + float(rng.normal(0, 0.5)), 0, len(RISK_BANDS) - 1))
        created = _created_at(rng, eco.config, cfg, history_days=1100.0)
        customer = eco.add_customer(
            created_at=created,
            country=code,
            region=region,
            home_lat=lat,
            home_lon=lon,
            segment=segment,
            spending_profile=profile,
            risk_band=RISK_BANDS[band_idx],
            is_synthetic_identity=False,
        )

        f_lo, f_hi = seg_freq[segment]
        a_lo, a_hi = profile_amount[profile]
        n_accounts = int(rng.integers(lo_acc, hi_acc + 1))
        for _a in range(n_accounts):
            amount_mean = float(rng.uniform(a_lo, a_hi))
            status = "active"
            u = float(rng.random())
            if u > 0.965:
                status = "dormant"
            elif u > 0.955:
                status = "closed"
            # An account opens after its holder joins, but never after the dataset
            # ends — an account created past ``config.end`` could hold no
            # transactions at all. A customer who joins late opens their first
            # account within days, as people actually do.
            opened = customer.created_at + timedelta(days=float(rng.random()) * 200.0)
            if opened >= eco.config.end:
                opened = customer.created_at + timedelta(
                    hours=float(rng.random()) * 72.0
                )
            opened = min(max(opened, customer.created_at), eco.config.end)
            eco.add_account(
                customer_id=customer.customer_id,
                created_at=opened,
                account_type=ACCOUNT_TYPES[int(rng.integers(0, len(ACCOUNT_TYPES)))],
                status=status,
                baseline_tx_frequency=round(float(rng.uniform(f_lo, f_hi)), 4),
                baseline_amount_mean=round(amount_mean, 2),
                baseline_amount_std=round(amount_mean * cv, 2),
            )


def _generate_beneficiaries(eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig) -> None:
    """Beneficiaries, some linked back to real accounts.

    The account link is what allows money to return to its origin, making
    circular-flow attacks possible — and it also occurs legitimately (paying
    your own savings account), so the link alone is not suspicious.
    """
    rng = seeds.rng("entities/beneficiaries")
    account_ids = [a.account_id for a in eco.accounts]
    for _ in range(cfg.n_beneficiaries):
        linked: Optional[str] = None
        if account_ids and rng.random() < 0.45:
            linked = account_ids[int(rng.integers(0, len(account_ids)))]
        country_idx = int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
        code, _r, _la, _lo = _pick_location(rng, country_idx)
        eco.add_beneficiary(
            created_at=_created_at(rng, eco.config, cfg, history_days=500.0),
            beneficiary_type=BENEFICIARY_TYPES[int(rng.integers(0, len(BENEFICIARY_TYPES)))],
            country=code,
            linked_account_id=linked,
        )


def _assign_infrastructure(eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig) -> None:
    """Wire accounts to devices and IPs, including *legitimate* sharing.

    This is the function that makes the graph benchmark honest. Three sources of
    benign overlap are created:

    1. **Households** — a set of customers share one or two devices, as a family
       or flatmates would.
    2. **Shared IP pools** — NAT / CGNAT / corporate egress addresses used by
       many unrelated accounts.
    3. **Incidental reuse** — devices are drawn from a finite pool, so some
       unrelated accounts coincidentally overlap.

    Consequently "many accounts on one device" and "many accounts on one IP" are
    both *normal* in this dataset, and a detector must find something stronger
    to distinguish a ring.
    """
    rng = seeds.rng("entities/infrastructure")
    devices = eco.devices
    ips = eco.ips
    warmup_start = eco.config.warmup_start
    shared_ips = [i.ip_id for i in ips if i.is_shared_legitimate]
    private_ips = [i.ip_id for i in ips if not i.is_shared_legitimate]
    # Entities created inside the observation window cannot be part of an account's
    # opening infrastructure — they do not exist yet. They are adopted later, below.
    established_device_idx = [
        i for i, d in enumerate(devices) if d.created_at < warmup_start
    ]
    new_device_ids = [d.device_id for d in devices if d.created_at >= warmup_start]
    new_private_ips = [
        i.ip_id
        for i in ips
        if i.created_at >= warmup_start and not i.is_shared_legitimate
    ]
    ip_created = {i.ip_id: i.created_at for i in ips}
    device_created = {d.device_id: d.created_at for d in devices}

    # --- households: mark a subset of devices as legitimately shared ---
    customers = eco.customers
    n_household_customers = int(len(customers) * cfg.household_fraction)
    household_members = list(range(n_household_customers))
    rng.shuffle(household_members)
    lo_h, hi_h = cfg.household_size
    device_cursor = 0
    households: List[List[str]] = []
    idx = 0
    while idx < len(household_members):
        size = int(rng.integers(lo_h, hi_h + 1))
        group = household_members[idx : idx + size]
        idx += size
        if len(group) < 2:
            break
        n_shared_devices = 1 if rng.random() < 0.7 else 2
        shared_device_ids: List[str] = []
        for _ in range(n_shared_devices):
            if device_cursor >= len(established_device_idx):
                break
            pos = established_device_idx[device_cursor]
            dev = devices[pos]
            device_cursor += 1
            # Replace with a shared-flagged copy in place.
            eco.devices[pos] = Device(
                device_id=dev.device_id,
                created_at=dev.created_at,
                device_type=dev.device_type,
                os_family=dev.os_family,
                fingerprint_hash=dev.fingerprint_hash,
                is_shared_legitimate=True,
            )
            shared_device_ids.append(dev.device_id)
        if not shared_device_ids:
            break
        households.append(shared_device_ids)
        for customer_pos in group:
            customer = customers[customer_pos]
            for account_id in eco.customer_accounts.get(customer.customer_id, []):
                eco.account_devices[account_id].extend(shared_device_ids)

    # --- per-account private devices and IPs ---
    remaining_devices = [
        devices[i].device_id for i in established_device_idx[device_cursor:]
    ]
    if not remaining_devices:  # pragma: no cover - defensive
        remaining_devices = [d.device_id for d in devices]
    established_private_ips = (
        [ip_id for ip_id in private_ips if ip_created[ip_id] < warmup_start]
        or private_ips
        or [i.ip_id for i in ips]
    )

    lo_u, hi_u = cfg.shared_ip_users
    shared_ip_capacity: Dict[str, int] = {
        ip_id: int(rng.integers(lo_u, hi_u + 1)) for ip_id in shared_ips
    }

    for account in eco.accounts:
        aid = account.account_id
        n_own = 1 if rng.random() < 0.78 else 2
        for _ in range(n_own):
            dev_id = remaining_devices[int(rng.integers(0, len(remaining_devices)))]
            if dev_id not in eco.account_devices[aid]:
                eco.account_devices[aid].append(dev_id)

        n_ips = 1 if rng.random() < 0.7 else 2
        for _ in range(n_ips):
            pool = established_private_ips
            ip_id = pool[int(rng.integers(0, len(pool)))]
            if ip_id not in eco.account_ips[aid]:
                eco.account_ips[aid].append(ip_id)
        # Some accounts sit behind a shared egress point.
        available = [k for k, v in shared_ip_capacity.items() if v > 0]
        if available and rng.random() < 0.30:
            ip_id = available[int(rng.integers(0, len(available)))]
            shared_ip_capacity[ip_id] -= 1
            if ip_id not in eco.account_ips[aid]:
                eco.account_ips[aid].append(ip_id)

        # --- mid-timeline adoption: the new handset, the rotated address ---
        # Without this, every device and IP that first appears after TRAIN was
        # created by an attack, and "unseen in TRAIN" becomes a fraud oracle —
        # measured at precision 0.956 for ``ip_unseen_in_train`` against account
        # takeover before this block existed.
        if new_device_ids and rng.random() < cfg.new_device_adoption_prob:
            dev_id = new_device_ids[int(rng.integers(0, len(new_device_ids)))]
            adopted = device_created[dev_id] + timedelta(
                days=float(rng.random()) * 20.0
            )
            if adopted < eco.config.end and dev_id not in eco.account_devices[aid]:
                eco.account_devices[aid].append(dev_id)
                eco.account_device_adoption[aid] = (dev_id, adopted)
        if new_private_ips and rng.random() < cfg.new_ip_rotation_prob:
            ip_id = new_private_ips[int(rng.integers(0, len(new_private_ips)))]
            rotated = ip_created[ip_id] + timedelta(days=float(rng.random()) * 10.0)
            if rotated < eco.config.end and ip_id not in eco.account_ips[aid]:
                eco.account_ips[aid].append(ip_id)
                eco.account_ip_adoption[aid] = (ip_id, rotated)

        if not eco.account_devices[aid]:  # pragma: no cover - defensive
            eco.account_devices[aid].append(remaining_devices[0])
        if not eco.account_ips[aid]:  # pragma: no cover - defensive
            eco.account_ips[aid].append(established_private_ips[0])


def _assign_payees(eco: Ecosystem, seeds: SeedRegistry, cfg: EcosystemConfig) -> None:
    """Give each account a small, stable payee list.

    The counterpart of :func:`_assign_infrastructure` for beneficiaries, and it
    exists for the same reason: a relationship that every legitimate account has
    with every legitimate entity is not a relationship, and its *absence* becomes a
    label. Normal transfers used to draw a payee uniformly from the whole pool, so a
    legitimate account accumulated as many distinct payees as it made transfers
    while a ring account paid the one or two beneficiaries its ring had registered.
    The audit measured ``beneficiary_account_degree`` at AUC 0.9889 against
    ``fake_merchant_ring``, 0.9683 against ``any_fraud``, and
    ``account_beneficiary_degree`` at 0.9846 against
    ``synthetic_identity_cluster`` — all three purely artefacts of that shortcut.

    Two pools reproduce the real shape. Most payees are **private**, reachable from
    one or two accounts, which puts legitimate payee degree in the same range a ring
    occupies. A minority are **popular** — utilities, subscriptions, tax authorities
    — so "many unrelated accounts pay this beneficiary" is also ordinary, exactly as
    shared NAT egress makes a busy IP ordinary.

    An account's own linked beneficiary is always on its list: paying your own
    savings account is the legitimate form of the link that makes circular flow
    possible, so the link must not be unique to fraud.
    """
    rng = seeds.rng("entities/payees")
    beneficiaries = eco.beneficiaries
    if not beneficiaries or not eco.accounts:  # pragma: no cover - defensive
        return

    n_popular = max(1, int(round(len(beneficiaries) * cfg.popular_beneficiary_fraction)))
    order = rng.permutation(len(beneficiaries))
    popular = [beneficiaries[int(i)].beneficiary_id for i in order[:n_popular]]
    private = [beneficiaries[int(i)].beneficiary_id for i in order[n_popular:]]
    if not private:  # pragma: no cover - defensive
        private = popular

    # Private payees are handed out from a shuffled queue rather than sampled with
    # replacement, so the "reachable from one or two accounts" property holds by
    # construction instead of on average. The queue wraps if accounts outnumber it.
    cursor = 0
    lo, hi = cfg.beneficiaries_per_account
    linked_for: Dict[str, List[str]] = {}
    for beneficiary in beneficiaries:
        if beneficiary.linked_account_id is not None:
            linked_for.setdefault(beneficiary.linked_account_id, []).append(
                beneficiary.beneficiary_id
            )

    for account in eco.accounts:
        payees: List[str] = []
        # Your own savings account, if the population happened to link one to you.
        for own in linked_for.get(account.account_id, ())[:1]:
            payees.append(own)
        n_payees = int(rng.integers(lo, hi + 1))
        while len(payees) < n_payees:
            if float(rng.random()) < cfg.popular_beneficiary_prob:
                candidate = popular[int(rng.integers(0, len(popular)))]
            else:
                candidate = private[cursor % len(private)]
                cursor += 1
            if candidate not in payees:
                payees.append(candidate)
        eco.account_beneficiaries[account.account_id] = payees


def generate_ecosystem(config: SimulationConfig, seeds: SeedRegistry) -> Ecosystem:
    """Build the full legitimate population.

    Order is fixed and each step draws from its own named RNG stream, so the
    result depends only on the seed and the config — not on call ordering.
    """
    eco = Ecosystem(config=config, minter=IdMinter(config.seed))
    cfg = config.ecosystem
    _generate_devices(eco, seeds, cfg)
    _generate_ips(eco, seeds, cfg)
    _generate_merchants(eco, seeds, cfg)
    _generate_customers_and_accounts(eco, seeds, cfg)
    _generate_beneficiaries(eco, seeds, cfg)
    _assign_infrastructure(eco, seeds, cfg)
    _assign_payees(eco, seeds, cfg)
    return eco


def shuffle_entity_tables(eco: Ecosystem, seeds: SeedRegistry) -> None:
    """Permute every entity table so that *row position* carries no information.

    The adversarial audit (:mod:`simulator.shortcut_audit`) measured
    ``merchant_creation_order_index`` at AUC 1.0000 against ``fake_merchant_ring``
    and five sibling features between 0.9837 and 1.0000 — six near-perfect
    oracles. The cause was structural rather than behavioural: the legitimate
    population is built first and every attack family *appends* to the same lists,
    so an entity's index in ``eco.merchants`` said exactly whether an attack
    created it. Benign household devices had the mirror-image problem, because
    :func:`_assign_infrastructure` flags the devices it walks *first*, making a low
    index mean ``is_shared_legitimate``.

    Row order in an entity table is not a real-world observable. A production
    database dump is not sorted "legitimate first, fraudulent second", so any model
    reading signal out of position is reading a generator artifact, not a property
    of fraud. The fix is therefore to destroy the ordering rather than to change
    anything about the entities: each table is permuted with its own named RNG
    stream, so the result still depends only on the seed and the config.

    Call this only after *all* generation is finished. Two attack families locate
    their own past output by slicing past the configured legitimate population size
    (``eco.beneficiaries[n_beneficiaries:]``), and that slice is meaningful only
    while creation order is intact.
    """
    rng = seeds.rng("entities/table_order")
    for name in ("customers", "merchants", "accounts", "devices", "ips", "beneficiaries"):
        table = getattr(eco, name)
        if len(table) < 2:
            continue
        order = rng.permutation(len(table))
        setattr(eco, name, [table[int(i)] for i in order])


def sharing_statistics(eco: Ecosystem) -> Dict[str, float]:
    """Summarise benign sharing — used by validation and the CP1.5 audit."""
    device_users: Dict[str, int] = {}
    ip_users: Dict[str, int] = {}
    for account_id, device_ids in eco.account_devices.items():
        for d in device_ids:
            device_users[d] = device_users.get(d, 0) + 1
    for account_id, ip_ids in eco.account_ips.items():
        for i in ip_ids:
            ip_users[i] = ip_users.get(i, 0) + 1
    multi_device = sum(1 for v in device_users.values() if v > 1)
    multi_ip = sum(1 for v in ip_users.values() if v > 1)
    return {
        "devices_with_multiple_accounts": float(multi_device),
        "ips_with_multiple_accounts": float(multi_ip),
        "max_accounts_per_device": float(max(device_users.values()) if device_users else 0),
        "max_accounts_per_ip": float(max(ip_users.values()) if ip_users else 0),
        "mean_devices_per_account": float(
            np.mean([len(v) for v in eco.account_devices.values()]) if eco.account_devices else 0
        ),
    }
