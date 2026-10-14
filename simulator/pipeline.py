"""Dataset assembly pipeline.

Ties the generators together into one reproducible dataset and enforces the two
properties that make the result usable as a benchmark.

**The temporal split.** Boundaries are the 60th and 80th percentiles of the
*normal* transaction timestamps inside the evaluation window, so each split holds
roughly its configured share of traffic regardless of seasonality. Splits are
half-open ``[lo, hi)``, which means a transaction landing exactly on a boundary
falls into the **later** split — the rule stated in
``docs/DATA_SPLIT_PROTOCOL.md`` §3, applied here rather than left to chance. Every
attack is fully contained within one split's window (see
:func:`simulator.attacks.base.attack_window`), so no ring is ever cut in half by a
boundary.

Warm-up traffic (``config.warmup_days`` before the evaluation window) is labelled
``"warmup"``, not ``"train"``. It exists so that 7-day rolling features have real
history on the very first evaluation day, and it is never scored.

**The prevalence budget.** ``AttackMix`` counts are *proportions*, not absolute
targets. The pipeline computes how many fraudulent transactions the configured
prevalence implies, then injects attack instances following the mix proportions
until that budget is met. This is what allows the same config to produce 1%, 5%,
10% or 25% prevalence without rewriting the attack mix. Every family is injected
at least once per split before proportional filling begins, so no split is ever
missing a family (``docs/ML_READINESS.md`` §5).

Realised prevalence will not land exactly on target — the last attack injected
overshoots by however many transactions it happens to contain. Both the target and
the realised value are recorded in the manifest, and
``config.prevalence_tolerance`` bounds the acceptable relative error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from simulator.attacks import generate_attack
from simulator.attacks.base import AttackContext
from simulator.base_generator import SeedRegistry, TimeWindow
from simulator.config import (
    SPLIT_NAMES,
    SimulationConfig,
    config_fingerprint,
    default_config,
)
from simulator.entities import (
    Ecosystem,
    generate_ecosystem,
    sharing_statistics,
    shuffle_entity_tables,
)
from simulator.ground_truth import (
    ATTACK_FAMILIES,
    DETECTION_CLASS,
    AttackRecord,
    GroundTruthRegistry,
)
from simulator.transactions import (
    TRANSACTION_FIELDS,
    Transaction,
    generate_normal_transactions,
)

__all__ = [
    "GeneratedDataset",
    "SplitBoundaries",
    "compute_boundaries",
    "fraud_budget",
    "generate_dataset",
]

WARMUP_SPLIT = "warmup"


# --------------------------------------------------------------------------
# Temporal split
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitBoundaries:
    """The two instants that divide the evaluation window into three splits."""

    evaluation_start: datetime
    train_end: datetime
    validation_end: datetime
    evaluation_end: datetime

    def __post_init__(self) -> None:
        ordered = (
            self.evaluation_start,
            self.train_end,
            self.validation_end,
            self.evaluation_end,
        )
        if list(ordered) != sorted(ordered):
            raise ValueError(f"split boundaries must be ordered, got {ordered}")

    def split_for(self, ts: datetime) -> str:
        """Assign a timestamp to its split.

        Half-open intervals throughout, so a transaction exactly on a boundary
        belongs to the later split — never duplicated, never dropped.
        """
        if ts < self.evaluation_start:
            return WARMUP_SPLIT
        if ts < self.train_end:
            return "train"
        if ts < self.validation_end:
            return "validation"
        return "heldout"

    def window(self, split: str) -> TimeWindow:
        if split == "train":
            return TimeWindow(self.evaluation_start, self.train_end)
        if split == "validation":
            return TimeWindow(self.train_end, self.validation_end)
        if split == "heldout":
            return TimeWindow(self.validation_end, self.evaluation_end)
        raise KeyError(f"no window for split {split!r}")

    def to_dict(self) -> Dict[str, str]:
        return {
            "evaluation_start": self.evaluation_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
        }


def compute_boundaries(
    normal: Sequence[Transaction], config: SimulationConfig
) -> SplitBoundaries:
    """Derive split boundaries from quantiles of in-window normal traffic.

    Quantiles of observed timestamps rather than a naive calendar split: activity
    is seasonal, so cutting the calendar in thirds would not give each split its
    configured *share of transactions*. Only normal traffic is used, because
    attacks have not been injected yet — and because letting fraud placement move
    the boundaries would make the split depend on the attack mix.
    """
    in_window = [
        t.timestamp for t in normal if config.start <= t.timestamp < config.end
    ]
    if len(in_window) < 3:  # pragma: no cover - defensive
        raise ValueError("not enough in-window transactions to form a split")
    in_window.sort()
    train_frac, val_frac, _held = config.split_fractions
    epochs = np.array([t.timestamp() for t in in_window], dtype=float)
    train_cut = float(np.quantile(epochs, train_frac))
    val_cut = float(np.quantile(epochs, train_frac + val_frac))
    tz = config.start.tzinfo
    return SplitBoundaries(
        evaluation_start=config.start,
        train_end=datetime.fromtimestamp(train_cut, tz=tz),
        validation_end=datetime.fromtimestamp(val_cut, tz=tz),
        evaluation_end=config.end,
    )


# --------------------------------------------------------------------------
# Prevalence budgeting
# --------------------------------------------------------------------------


def fraud_budget(n_normal: int, prevalence: float) -> int:
    """Fraudulent transactions needed so fraud/(normal+fraud) == prevalence.

    Solving ``f / (n + f) = p`` gives ``f = p*n / (1 - p)``. Note this is a count
    of *rows*, not of attacks: one microtransaction attack contributes hundreds of
    rows while one takeover contributes a handful.
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    return int(round(prevalence * n_normal / (1.0 - prevalence)))


def family_schedule(mix: Dict[str, int]) -> List[str]:
    """Interleave families in proportion to their mix counts.

    Round-robin rather than blocked (``A A B B C``…), so that truncating the
    schedule part-way — which happens whenever the budget runs out mid-cycle —
    still leaves roughly the intended family proportions instead of over-weighting
    whichever family happened to come first.
    """
    remaining = {f: max(0, int(mix.get(f, 0))) for f in ATTACK_FAMILIES}
    order: List[str] = []
    while any(remaining.values()):
        for family in ATTACK_FAMILIES:
            if remaining[family] > 0:
                order.append(family)
                remaining[family] -= 1
    return order


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass
class GeneratedDataset:
    """One complete synthetic dataset plus the metadata to reproduce it."""

    config: SimulationConfig
    ecosystem: Ecosystem
    transactions: List[Transaction]
    ground_truth: GroundTruthRegistry
    boundaries: SplitBoundaries
    split_of: Dict[str, str] = field(default_factory=dict)
    """transaction_id -> split name (including ``"warmup"``)."""
    manifest: Dict[str, object] = field(default_factory=dict)

    def split_counts(self) -> Dict[str, Dict[str, int]]:
        """Per-split totals and realised prevalence."""
        out: Dict[str, Dict[str, int]] = {}
        for name in (WARMUP_SPLIT,) + tuple(SPLIT_NAMES):
            rows = [t for t in self.transactions if self.split_of[t.transaction_id] == name]
            fraud = sum(1 for t in rows if t.is_fraud)
            out[name] = {
                "transactions": len(rows),
                "fraud": fraud,
                "normal": len(rows) - fraud,
            }
        return out


def _activity_index(transactions: Sequence[Transaction]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for txn in transactions:
        counts[txn.account_id] = counts.get(txn.account_id, 0) + 1
    return counts


def _inject_split(
    *,
    eco: Ecosystem,
    seeds: SeedRegistry,
    config: SimulationConfig,
    split: str,
    window: TimeWindow,
    budget: int,
    activity: Dict[str, int],
    merchant_activity: Dict[str, int],
) -> Tuple[List[Transaction], List[Transaction], List[AttackRecord]]:
    """Inject attacks into one split until its fraud budget is met.

    Returns ``(fraud_rows, cover_rows, records)``. Cover rows are legitimate
    traffic a family needed the ecosystem to contain — see
    :attr:`simulator.attacks.base.AttackResult.cover_transactions`. They are
    returned separately so they count toward neither the fraud budget nor the
    realised prevalence numerator.
    """
    split_cfg = config.split_config(split)
    schedule = family_schedule(split_cfg.mix.as_mapping())
    if not schedule:  # pragma: no cover - defensive
        return [], [], []

    transactions: List[Transaction] = []
    cover: List[Transaction] = []
    records: List[AttackRecord] = []
    instance_counter: Dict[str, int] = {f: 0 for f in ATTACK_FAMILIES}
    # Accounts already taken over / spiked in this split. A single account being
    # seized five times in one window is not credible, so Class A victims are
    # not reused; Class B membership may legitimately overlap.
    class_a_victims: set = set()

    # Fraud rows already placed on each merchant *in this split*, accumulated across
    # instances. ``fake_merchant_ring`` recycles shells, so no single instance can
    # see a shell's total exposure; without a shared tally each one would dilute
    # only its own contribution and the shell's actual fraud share would drift up
    # with every ring that reused it.
    merchant_fraud_rows: Dict[str, int] = {}

    # Guarantee coverage first: one instance of every family, before any
    # proportional filling, so a tight budget can never drop a family entirely.
    guaranteed = list(ATTACK_FAMILIES)
    emitted = 0
    position = 0
    max_instances = max(len(guaranteed), budget)  # hard stop; budget is in rows

    while True:
        if position < len(guaranteed):
            family = guaranteed[position]
        else:
            if emitted >= budget:
                break
            family = schedule[(position - len(guaranteed)) % len(schedule)]
        position += 1
        if position > max_instances + len(guaranteed):  # pragma: no cover
            break

        index = instance_counter[family]
        instance_counter[family] = index + 1
        attack_id = f"atk_{split}_{family}_{index:04d}"
        is_class_a = DETECTION_CLASS[family] == "A"
        ctx = AttackContext(
            eco=eco,
            rng=seeds.rng(f"attack/{split}/{family}/{index}"),
            split=split,
            window=window,
            attack_id=attack_id,
            account_activity=activity,
            merchant_activity=merchant_activity,
            merchant_fraud_rows=merchant_fraud_rows,
            exclude_accounts=set(class_a_victims) if is_class_a else set(),
        )
        result = generate_attack(family, ctx, split_cfg.params_for(family))
        if not result.transactions:  # pragma: no cover - defensive
            continue
        if is_class_a:
            class_a_victims.update(result.record.affected_account_ids)
        transactions.extend(result.transactions)
        cover.extend(result.cover_transactions)
        records.append(result.record)
        emitted += len(result.transactions)

    return transactions, cover, records


def generate_dataset(config: Optional[SimulationConfig] = None) -> GeneratedDataset:
    """Build the complete dataset for ``config``.

    The order of operations is fixed and every stage draws from its own named RNG
    stream, so the output depends only on the seed and the config.
    """
    config = config or default_config()
    seeds = SeedRegistry(config.seed)

    eco = generate_ecosystem(config, seeds)
    normal = generate_normal_transactions(eco, seeds)
    boundaries = compute_boundaries(normal, config)

    # Budget is apportioned across splits in proportion to their normal traffic,
    # so each split lands near the same prevalence rather than the dataset only
    # hitting the target in aggregate.
    normal_by_split: Dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    # Per-split organic purchase counts per merchant. ``fake_merchant_ring`` reads
    # its own split's copy to decide how much top-up cover a shell needs, and
    # mutates it as it injects, so instances that recycle the same shell see what
    # earlier ones already added. Kept per split because the per-merchant fraud rate
    # a later checkpoint could learn is fitted on TRAIN alone — diluting a shell's
    # history in the split where its ring rows actually land is what matters, and
    # confining the top-up to that split is what stops cover traffic from bleeding
    # backwards and skewing the split fractions.
    merchant_normal: Dict[str, Dict[str, int]] = {name: {} for name in SPLIT_NAMES}
    for txn in normal:
        split = boundaries.split_for(txn.timestamp)
        if split in normal_by_split:
            normal_by_split[split] += 1
            if txn.merchant_id is not None:
                per_merchant = merchant_normal[split]
                per_merchant[txn.merchant_id] = per_merchant.get(txn.merchant_id, 0) + 1

    activity = _activity_index(normal)
    fraud_transactions: List[Transaction] = []
    cover_transactions: List[Transaction] = []
    records: List[AttackRecord] = []
    budgets: Dict[str, int] = {}
    for split in SPLIT_NAMES:
        budgets[split] = fraud_budget(normal_by_split[split], config.fraud_prevalence)
        split_txns, split_cover, split_records = _inject_split(
            eco=eco,
            seeds=seeds,
            config=config,
            split=split,
            window=boundaries.window(split),
            budget=budgets[split],
            activity=activity,
            merchant_activity=merchant_normal[split],
        )
        fraud_transactions.extend(split_txns)
        cover_transactions.extend(split_cover)
        records.extend(split_records)

    # Destroy entity-table row order *after* every generator has finished, because
    # two attack families locate their own past output by slicing past the
    # configured legitimate population size. Until this call, position in
    # ``eco.merchants`` / ``eco.beneficiaries`` etc. encoded provenance:
    # "created by an attack" is simply "index >= n_legitimate". The audit measured
    # merchant_creation_order_index at AUC 1.0000 on that basis. Row order is not a
    # real-world observable, so removing it is a fix to the generating process, not
    # a weakening of the data (``docs/ML_READINESS.md`` §3).
    shuffle_entity_tables(eco, seeds)

    registry = GroundTruthRegistry(records)
    # Cover rows join the *normal* pool: they are legitimate traffic that merely
    # happens to touch attack-created entities, so they belong with normal traffic
    # for every purpose — prevalence, labels, and ground truth alike.
    transactions = list(normal) + cover_transactions + fraud_transactions
    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    split_of = {t.transaction_id: boundaries.split_for(t.timestamp) for t in transactions}

    dataset = GeneratedDataset(
        config=config,
        ecosystem=eco,
        transactions=transactions,
        ground_truth=registry,
        boundaries=boundaries,
        split_of=split_of,
    )
    dataset.manifest = build_manifest(dataset, budgets=budgets, seeds=seeds)
    return dataset


def build_manifest(
    dataset: GeneratedDataset,
    *,
    budgets: Dict[str, int],
    seeds: SeedRegistry,
) -> Dict[str, object]:
    """Everything needed to reproduce and audit the dataset.

    Includes a content hash of the transaction rows: two runs claiming to be
    identical can be compared on one string instead of diffing the whole file.
    """
    counts = dataset.split_counts()
    total = len(dataset.transactions)
    total_fraud = sum(1 for t in dataset.transactions if t.is_fraud)
    realised = {
        name: (c["fraud"] / c["transactions"] if c["transactions"] else 0.0)
        for name, c in counts.items()
    }
    return {
        "seed": dataset.config.seed,
        "config_fingerprint": config_fingerprint(dataset.config),
        "config": dataset.config.to_dict(),
        "boundaries": dataset.boundaries.to_dict(),
        "entity_counts": dataset.ecosystem.entity_counts(),
        "sharing_statistics": sharing_statistics(dataset.ecosystem),
        "transactions": {
            "total": total,
            "fraud": total_fraud,
            "normal": total - total_fraud,
            "by_split": counts,
        },
        "prevalence": {
            "configured": dataset.config.fraud_prevalence,
            "realised_overall": (total_fraud / total if total else 0.0),
            "realised_by_split": realised,
            "row_budget_by_split": budgets,
            "tolerance": dataset.config.prevalence_tolerance,
        },
        "attacks": {
            "total": len(dataset.ground_truth),
            "by_family": dataset.ground_truth.family_counts(),
            "fraud_rows_by_family": dataset.ground_truth.transaction_counts_by_family(),
            "coverage": dataset.ground_truth.coverage_summary(),
        },
        "content_hash": content_hash(dataset),
        "rng_streams": len(seeds.issued_paths()),
    }


def content_hash(dataset: GeneratedDataset) -> str:
    """SHA256 over the canonical transaction rows.

    Deliberately covers the label columns too: a run that produced the same rows
    with different labels is *not* the same dataset.
    """
    digest = hashlib.sha256()
    for txn in dataset.transactions:
        row = "|".join(_field_repr(getattr(txn, name)) for name in TRANSACTION_FIELDS)
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _field_repr(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
