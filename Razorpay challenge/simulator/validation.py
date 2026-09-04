"""Dataset validation — structural integrity plus ML-readiness properties.

This module is **evaluation-side infrastructure**, in the same category as
:mod:`simulator.ground_truth`: it is allowed to read labels because its job is to
audit the dataset, and it is never imported by feature, model, or scoring code
(``docs/FEATURE_LEAKAGE_POLICY.md`` §4).

Checks fall into two groups, and the distinction matters.

*Integrity* checks are hard requirements. Every foreign key resolves, IDs are
unique, timestamps sit inside their split, warm-up carries no fraud, every
fraudulent row is attributable to exactly one attack. A failure here means the
dataset is malformed and nothing downstream can be trusted, so these are errors.

*Readiness* checks are about whether the dataset poses a real problem. Fraud must
not be separable by any single obvious rule — not by amount, not by transaction
count, not by "the device is shared". These are the properties that make a fraud
benchmark meaningful rather than a lookup table, and they are asserted in
``docs/ML_READINESS.md`` §3. Some are reported as warnings rather than errors
because they are statistical rather than absolute: the honest response to a
borderline overlap statistic is to surface the number, not to fail the build.

Every check records the *measurement* it made, not just a verdict, so the
checkpoint report can quote figures instead of "all tests passed".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from simulator.config import SPLIT_NAMES
from simulator.ground_truth import (
    ATTACK_FAMILIES,
    DETECTION_CLASS,
    validate_record_completeness,
)
from simulator.pipeline import WARMUP_SPLIT, GeneratedDataset
from simulator.transactions import CHANNELS, STATUSES, TRANSACTION_TYPES

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_dataset",
]

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)

    def error(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(ERROR, code, message))

    def warn(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(WARNING, code, message))

    def measure(self, key: str, value: Any) -> None:
        self.measurements[key] = value

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True when there are no errors. Warnings do not fail the dataset."""
        return not self.errors

    def summary(self) -> str:
        return (
            f"{'PASS' if self.ok else 'FAIL'} — "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [{"code": i.code, "message": i.message} for i in self.errors],
            "warnings": [{"code": i.code, "message": i.message} for i in self.warnings],
            "measurements": self.measurements,
        }


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------


def _check_referential_integrity(ds: GeneratedDataset, rep: ValidationReport) -> None:
    eco = ds.ecosystem
    accounts = eco.account_by_id
    customers = eco.customer_by_id
    merchants = eco.merchant_by_id
    devices = eco.device_by_id
    ips = eco.ip_by_id
    beneficiaries = eco.beneficiary_by_id

    dangling: Counter = Counter()
    mismatched_customer = 0
    both_targets = 0
    neither_target = 0

    for txn in ds.transactions:
        if txn.account_id not in accounts:
            dangling["account"] += 1
        elif accounts[txn.account_id].customer_id != txn.customer_id:
            # The row's customer must be the account's owner, or joins built on
            # either key would silently disagree.
            mismatched_customer += 1
        if txn.customer_id not in customers:
            dangling["customer"] += 1
        if txn.device_id not in devices:
            dangling["device"] += 1
        if txn.ip_id not in ips:
            dangling["ip"] += 1
        if txn.merchant_id is not None and txn.merchant_id not in merchants:
            dangling["merchant"] += 1
        if txn.beneficiary_id is not None and txn.beneficiary_id not in beneficiaries:
            dangling["beneficiary"] += 1
        # A transaction targets a merchant or a beneficiary, never both.
        if txn.merchant_id is not None and txn.beneficiary_id is not None:
            both_targets += 1
        if txn.merchant_id is None and txn.beneficiary_id is None:
            if txn.transaction_type not in ("withdrawal",):
                neither_target += 1

    rep.measure("dangling_references", dict(dangling))
    if dangling:
        rep.error(
            "referential_integrity",
            f"transactions reference missing entities: {dict(dangling)}",
        )
    if mismatched_customer:
        rep.error(
            "customer_account_mismatch",
            f"{mismatched_customer} rows whose customer_id is not the account owner",
        )
    if both_targets:
        rep.error(
            "ambiguous_target",
            f"{both_targets} rows carry both a merchant_id and a beneficiary_id",
        )
    if neither_target:
        rep.error(
            "missing_target",
            f"{neither_target} non-withdrawal rows have neither merchant nor beneficiary",
        )


def _check_ids_and_fields(ds: GeneratedDataset, rep: ValidationReport) -> None:
    ids = [t.transaction_id for t in ds.transactions]
    unique = set(ids)
    rep.measure("transaction_count", len(ids))
    rep.measure("unique_transaction_ids", len(unique))
    if len(unique) != len(ids):
        rep.error(
            "duplicate_transaction_ids",
            f"{len(ids) - len(unique)} duplicate transaction_id values",
        )

    # Entity IDs must be unique within their own type.
    eco = ds.ecosystem
    for label, seq in (
        ("customer", [c.customer_id for c in eco.customers]),
        ("merchant", [m.merchant_id for m in eco.merchants]),
        ("account", [a.account_id for a in eco.accounts]),
        ("device", [d.device_id for d in eco.devices]),
        ("ip", [i.ip_id for i in eco.ips]),
        ("beneficiary", [b.beneficiary_id for b in eco.beneficiaries]),
    ):
        if len(set(seq)) != len(seq):
            rep.error(
                "duplicate_entity_ids",
                f"{label}: {len(seq) - len(set(seq))} duplicate identifiers",
            )

    bad_amount = sum(1 for t in ds.transactions if not (t.amount > 0))
    if bad_amount:
        rep.error("nonpositive_amount", f"{bad_amount} rows with amount <= 0")

    # Amounts must look like money, not like float noise.
    unrounded = sum(
        1 for t in ds.transactions if abs(t.amount - round(t.amount, 2)) > 1e-9
    )
    if unrounded:
        rep.error("unrounded_amount", f"{unrounded} rows with >2 decimal places")

    currencies = {t.currency for t in ds.transactions}
    if currencies != {ds.config.currency}:
        rep.error("currency_mismatch", f"unexpected currencies: {sorted(currencies)}")

    bad_channel = {t.channel for t in ds.transactions} - set(CHANNELS)
    bad_status = {t.status for t in ds.transactions} - set(STATUSES)
    bad_type = {t.transaction_type for t in ds.transactions} - set(TRANSACTION_TYPES)
    for label, bad in (
        ("channel", bad_channel),
        ("status", bad_status),
        ("transaction_type", bad_type),
    ):
        if bad:
            rep.error("invalid_enum", f"unexpected {label} values: {sorted(bad)}")

    off_grid = sum(
        1 for t in ds.transactions if not (-90.0 <= t.lat <= 90.0 and -180.0 <= t.lon <= 180.0)
    )
    if off_grid:
        rep.error("coordinates_out_of_range", f"{off_grid} rows with invalid lat/lon")


def _check_temporal_split(ds: GeneratedDataset, rep: ValidationReport) -> None:
    b = ds.boundaries
    counts = ds.split_counts()
    rep.measure("split_counts", counts)
    rep.measure("boundaries", b.to_dict())

    # Warm-up exists and is fraud-free: 7-day features need history, and scoring
    # never touches it.
    if counts[WARMUP_SPLIT]["transactions"] == 0:
        rep.error("no_warmup", "warm-up window contains no transactions")
    if counts[WARMUP_SPLIT]["fraud"] != 0:
        rep.error(
            "fraud_in_warmup",
            f"{counts[WARMUP_SPLIT]['fraud']} fraudulent rows inside the warm-up window",
        )

    # Chronological isolation: no split may contain a timestamp belonging to another.
    for split in SPLIT_NAMES:
        window = b.window(split)
        rows = [t for t in ds.transactions if ds.split_of[t.transaction_id] == split]
        outside = [t for t in rows if not (window.start <= t.timestamp < window.end)]
        if outside:
            rep.error(
                "split_leakage",
                f"{len(outside)} rows assigned to {split} fall outside its window",
            )

    # Ordering: train strictly precedes validation strictly precedes heldout.
    maxima = {}
    minima = {}
    for split in SPLIT_NAMES:
        ts = [t.timestamp for t in ds.transactions if ds.split_of[t.transaction_id] == split]
        if not ts:
            rep.error("empty_split", f"{split} contains no transactions")
            continue
        minima[split], maxima[split] = min(ts), max(ts)
    if len(minima) == len(SPLIT_NAMES):
        if not (maxima["train"] < minima["validation"] < maxima["validation"] < minima["heldout"]):
            rep.error(
                "split_order",
                "splits are not strictly chronologically ordered "
                f"(train_max={maxima['train']}, val_min={minima['validation']}, "
                f"val_max={maxima['validation']}, heldout_min={minima['heldout']})",
            )

    # Realised split proportions should track the configured fractions.
    total = sum(counts[s]["transactions"] for s in SPLIT_NAMES)
    realised = tuple(counts[s]["transactions"] / total for s in SPLIT_NAMES)
    rep.measure("split_fractions_configured", list(ds.config.split_fractions))
    rep.measure("split_fractions_realised", [round(f, 4) for f in realised])
    for name, want, got in zip(SPLIT_NAMES, ds.config.split_fractions, realised):
        if abs(got - want) > 0.03:
            rep.warn(
                "split_fraction_drift",
                f"{name} holds {got:.3f} of transactions, configured {want:.3f}",
            )


#: Transaction field -> (ecosystem lookup attribute, human label). Every entity a
#: transaction references must already exist at the moment the transaction occurs.
_CAUSAL_REFERENCES = (
    ("account_id", "account_by_id", "account"),
    ("customer_id", "customer_by_id", "customer"),
    ("merchant_id", "merchant_by_id", "merchant"),
    ("device_id", "device_by_id", "device"),
    ("ip_id", "ip_by_id", "ip"),
    ("beneficiary_id", "beneficiary_by_id", "beneficiary"),
)


def _check_temporal_causality(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """No transaction may reference an entity that did not yet exist.

    This is the crudest form of future leakage and the easiest to introduce by
    accident: an attack recruits an account created later in the same split, or a
    ring transacts with a shell merchant before it was onboarded. The resulting row
    carries a *negative* entity age — the audit measured ``account_age_days`` at
    -108.2 days on exactly that path (see
    :func:`simulator.attacks.base.pick_accounts`). A negative age is not merely odd:
    it is information that could not have been observed at scoring time, and any
    model would exploit it.

    Reported as an **error**, not a warning. Unlike the statistical readiness checks,
    there is no borderline case here — either the reference is causally possible or
    the dataset is describing something that could not have happened.

    Also asserts that no entity is created after the dataset's own end, which would
    mean the generator invented history outside the observation window.
    """
    eco = ds.ecosystem
    lookups = {attr: getattr(eco, attr) for _, attr, _ in _CAUSAL_REFERENCES}
    violations: Dict[str, int] = {}
    worst: Dict[str, float] = {}

    for txn in ds.transactions:
        for field_name, attr, label in _CAUSAL_REFERENCES:
            entity_id = getattr(txn, field_name)
            if entity_id is None:
                continue
            entity = lookups[attr].get(entity_id)
            if entity is None:
                continue  # referential integrity reports the dangling key itself
            age_days = (txn.timestamp - entity.created_at).total_seconds() / 86400.0
            if age_days < 0.0:
                key = f"transaction_before_{label}_creation"
                violations[key] = violations.get(key, 0) + 1
                worst[key] = min(worst.get(key, 0.0), age_days)

    for key, count in sorted(violations.items()):
        rep.error(
            key,
            f"{count} transactions reference an entity created after the transaction "
            f"timestamp (most negative age {worst[key]:.1f} days)",
        )
    rep.measure(
        "causality_violations",
        {key: violations.get(key, 0) for _, _, label in _CAUSAL_REFERENCES
         for key in (f"transaction_before_{label}_creation",)},
    )

    # Entities may not be created after the observation window closes.
    horizon = ds.config.end
    late = {
        label: sum(1 for e in lookups[attr].values() if e.created_at > horizon)
        for _, attr, label in _CAUSAL_REFERENCES
    }
    rep.measure("entities_created_after_end", late)
    for label, count in late.items():
        if count:
            rep.error(
                "entity_created_after_dataset_end",
                f"{count} {label} entities have created_at after the dataset end {horizon}",
            )


def _check_prevalence(ds: GeneratedDataset, rep: ValidationReport) -> None:
    counts = ds.split_counts()
    target = ds.config.fraud_prevalence
    tol = ds.config.prevalence_tolerance

    evaluated = [counts[s] for s in SPLIT_NAMES]
    total = sum(c["transactions"] for c in evaluated)
    fraud = sum(c["fraud"] for c in evaluated)
    overall = fraud / total if total else 0.0
    rep.measure("prevalence_target", target)
    rep.measure("prevalence_realised_evaluation_window", round(overall, 5))
    rep.measure(
        "prevalence_realised_by_split",
        {
            s: round(counts[s]["fraud"] / counts[s]["transactions"], 5)
            for s in SPLIT_NAMES
            if counts[s]["transactions"]
        },
    )

    if target > 0 and abs(overall - target) / target > tol:
        rep.error(
            "prevalence_out_of_tolerance",
            f"realised prevalence {overall:.4f} deviates from target {target:.4f} "
            f"by more than the {tol:.0%} tolerance",
        )
    for split in SPLIT_NAMES:
        c = counts[split]
        if not c["transactions"]:
            continue
        got = c["fraud"] / c["transactions"]
        if target > 0 and abs(got - target) / target > tol:
            rep.warn(
                "split_prevalence_drift",
                f"{split} prevalence {got:.4f} vs target {target:.4f}",
            )


def _check_ground_truth(ds: GeneratedDataset, rep: ValidationReport) -> None:
    registry = ds.ground_truth
    rep.measure("attack_count", len(registry))
    rep.measure("attacks_by_family", registry.family_counts())
    rep.measure("fraud_rows_by_family", registry.transaction_counts_by_family())

    problems: List[str] = []
    for record in registry:
        problems.extend(f"{record.attack_id}: {p}" for p in validate_record_completeness(record))
    rep.measure("incomplete_ground_truth_records", len(problems))
    if problems:
        rep.error(
            "incomplete_ground_truth",
            f"{len(problems)} completeness problems, first: {problems[0]}",
        )

    # Labels and the registry must agree exactly, in both directions.
    gt_ids = registry.fraudulent_transaction_ids()
    labelled = {t.transaction_id for t in ds.transactions if t.is_fraud}
    if gt_ids != labelled:
        rep.error(
            "label_registry_disagreement",
            f"{len(labelled - gt_ids)} labelled rows missing from the registry, "
            f"{len(gt_ids - labelled)} registry rows not labelled",
        )
    stray = [t.transaction_id for t in ds.transactions if not t.is_fraud and t.attack_id]
    if stray:
        rep.error("attack_id_on_normal_row", f"{len(stray)} normal rows carry an attack_id")

    # Every family must appear in every split, or a family becomes unevaluable on
    # held-out data (docs/ML_READINESS.md §5).
    per_split: Dict[str, set] = {s: set() for s in SPLIT_NAMES}
    for record in registry:
        if record.split in per_split:
            per_split[record.split].add(record.attack_type)
    coverage = {s: sorted(per_split[s]) for s in SPLIT_NAMES}
    rep.measure("families_per_split", {s: len(v) for s, v in coverage.items()})
    for split in SPLIT_NAMES:
        missing = set(ATTACK_FAMILIES) - per_split[split]
        if missing:
            rep.error(
                "family_missing_from_split",
                f"{split} contains no instance of: {sorted(missing)}",
            )

    # Both detection classes must be present per split, since they are reported
    # separately (docs/EVALUATION.md §6).
    for split in SPLIT_NAMES:
        classes = {DETECTION_CLASS[f] for f in per_split[split]}
        if classes != {"A", "B"}:
            rep.error(
                "detection_class_missing",
                f"{split} lacks detection class(es): {sorted({'A','B'} - classes)}",
            )

    # Attacks must sit entirely inside the split they claim.
    for record in registry:
        if record.split not in SPLIT_NAMES:
            rep.error("unknown_attack_split", f"{record.attack_id} has split {record.split!r}")
            continue
        window = ds.boundaries.window(record.split)
        if not (window.start <= record.start_time and record.end_time < window.end):
            rep.error(
                "attack_crosses_boundary",
                f"{record.attack_id} spans outside its {record.split} window",
            )

    # Each fraudulent row's split must match its attack's split.
    mismatched = 0
    for txn in ds.transactions:
        if not txn.is_fraud or not txn.attack_id:
            continue
        try:
            record = registry.by_id(txn.attack_id)
        except KeyError:
            continue
        if ds.split_of[txn.transaction_id] != record.split:
            mismatched += 1
    if mismatched:
        rep.error(
            "row_attack_split_mismatch",
            f"{mismatched} fraudulent rows sit in a different split than their attack",
        )


# --------------------------------------------------------------------------
# ML readiness
# --------------------------------------------------------------------------


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, *, bins: int = 60) -> float:
    """Histogram overlap of two samples on a shared log scale, in [0, 1].

    0 means the distributions are disjoint — a threshold would separate them
    perfectly. Computed in log space because transaction amounts span orders of
    magnitude and a linear histogram would put almost everything in one bin.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    la, lb = np.log10(np.maximum(a, 0.01)), np.log10(np.maximum(b, 0.01))
    lo = min(la.min(), lb.min())
    hi = max(la.max(), lb.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(la, bins=edges)
    hb, _ = np.histogram(lb, bins=edges)
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    return float(np.minimum(pa, pb).sum())


def _best_band_f1(amounts: np.ndarray, target: np.ndarray, *, grid: int = 80) -> tuple:
    """Best F1 attainable by an amount *band* ``lo <= x <= hi``, and that band.

    Strictly stronger than a one-sided threshold: it catches families that live in
    a *middle* range, which a "greater than" rule would miss. The search uses the
    labels and sweeps every pair of grid quantiles, so it is deliberately generous
    to the adversary — an upper bound on what amount alone can achieve.

    Implemented on the sorted array with a prefix sum, so each candidate band costs
    O(1) and the whole sweep is O(n log n + grid²).
    """
    if amounts.size == 0 or not target.any():
        return 0.0, None
    order = np.argsort(amounts, kind="stable")
    sorted_amounts = amounts[order]
    labels = target[order].astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(labels)])
    total_pos = int(labels.sum())
    cuts = np.unique(np.linspace(0, len(labels), grid + 1).astype(int))
    best_f1, best_band = 0.0, None
    for a_i in range(len(cuts)):
        for b_i in range(a_i + 1, len(cuts)):
            lo_i, hi_i = int(cuts[a_i]), int(cuts[b_i])
            tp = int(prefix[hi_i] - prefix[lo_i])
            if tp == 0:
                continue
            precision = tp / (hi_i - lo_i)
            recall = tp / total_pos
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_band = (float(sorted_amounts[lo_i]), float(sorted_amounts[hi_i - 1]))
    return best_f1, best_band


def _check_amount_non_separability(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Amount alone must not be sufficient to detect fraud — overall or per family.

    The gate is a *classification* metric, not distributional overlap. That choice
    is deliberate. Fraudulent amounts legitimately differ in distribution from
    legitimate ones — laundering cycles move large round-ish sums, probing attacks
    move tiny ones — and demanding near-identical histograms would be unrealistic in
    the opposite direction. What ``docs/ML_READINESS.md`` §3 forbids is amount being
    *sufficient*, so the check asks the operational question directly: with oracle
    knowledge of the labels, how good a detector is the best possible amount band?

    Overlap is still measured and reported, and a very low value is still an error,
    because fully disjoint supports would mean a trivially perfect rule exists.
    """
    fraud = np.array([t.amount for t in ds.transactions if t.is_fraud])
    normal = np.array([t.amount for t in ds.transactions if not t.is_fraud])
    if fraud.size == 0 or normal.size == 0:
        rep.error("degenerate_labels", "dataset has only one class")
        return

    overlap = _overlap_coefficient(fraud, normal)
    rep.measure("amount_overlap_coefficient", round(overlap, 4))
    rep.measure(
        "amount_percentiles",
        {
            "fraud": {p: round(float(np.percentile(fraud, p)), 2) for p in (50, 90, 99)},
            "normal": {p: round(float(np.percentile(normal, p)), 2) for p in (50, 90, 99)},
        },
    )
    if overlap < 0.15:
        rep.error(
            "amount_supports_disjoint",
            f"fraud and normal amount distributions overlap only {overlap:.2f}; "
            "their supports are nearly disjoint",
        )

    amounts = np.array([t.amount for t in ds.transactions], dtype=float)
    labels = np.array([t.is_fraud for t in ds.transactions], dtype=bool)
    best_f1, best_band = _best_band_f1(amounts, labels)
    rep.measure("best_amount_band_f1", round(best_f1, 4))
    rep.measure(
        "best_amount_band",
        [round(b, 2) for b in best_band] if best_band else None,
    )
    if best_f1 > 0.5:
        rep.error(
            "amount_sufficient_overall",
            f"an oracle-chosen amount band reaches F1={best_f1:.3f}; fraud is too "
            "easily separable by amount alone",
        )
    elif best_f1 > 0.35:
        rep.warn(
            "amount_informative",
            f"best amount band reaches F1={best_f1:.3f}",
        )

    # Per family: an aggregate that looks safe can still hide one family sitting in
    # its own amount band. Measured before this check existed: "amount < 6" caught
    # 100% of microtransaction rows at 79% precision.
    family_of: Dict[str, str] = {}
    for record in ds.ground_truth:
        for tid in record.affected_transaction_ids:
            family_of[tid] = record.attack_type
    families = np.array([family_of.get(t.transaction_id, "") for t in ds.transactions])

    per_family: Dict[str, float] = {}
    offenders: List[str] = []
    for family in ATTACK_FAMILIES:
        mask = families == family
        if not mask.any():
            continue
        f1, band = _best_band_f1(amounts, mask)
        per_family[family] = round(f1, 4)
        if f1 > 0.6:
            offenders.append(f"{family} (F1={f1:.3f}, band={band})")
    rep.measure("best_amount_band_f1_by_family", per_family)
    if offenders:
        rep.error(
            "family_separable_by_amount",
            "these families are isolable by an amount band alone: " + "; ".join(offenders),
        )

    # Legitimate traffic must occupy both tails, so neither "large" nor "small" is
    # a synonym for "fraudulent".
    fraud_median = float(np.median(fraud))
    rep.measure("legit_rows_above_fraud_median", int((normal > fraud_median).sum()))
    rep.measure("legit_max_amount", round(float(normal.max()), 2))
    rep.measure("fraud_max_amount", round(float(fraud.max()), 2))
    small_cut = float(np.percentile(fraud, 5))
    legit_small = int((normal <= small_cut).sum())
    rep.measure("fraud_p5_amount", round(small_cut, 2))
    rep.measure("legit_rows_below_fraud_p5", legit_small)
    if int((normal > fraud_median).sum()) == 0:
        rep.error(
            "no_legitimate_outliers",
            "no legitimate transaction exceeds the fraud median amount",
        )
    if legit_small == 0:
        rep.error(
            "no_legitimate_small_spend",
            "no legitimate transaction is as small as the 5th percentile of fraud; "
            "'tiny amount' would imply fraud",
        )
    if float(normal.max()) < float(fraud.max()):
        rep.warn(
            "fraud_holds_global_max",
            f"largest transaction is fraudulent ({fraud.max():.2f} > {normal.max():.2f})",
        )


def _check_velocity_non_separability(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Per-account transaction counts must not separate the classes either.

    Class A attacks are bursts *relative to the victim's own baseline*, so a global
    "count > N" rule should fail. This checks that ordinary busy accounts are at
    least as active as attacked ones.
    """
    per_account_total: Counter = Counter()
    per_account_fraud: Counter = Counter()
    for txn in ds.transactions:
        per_account_total[txn.account_id] += 1
        if txn.is_fraud:
            per_account_fraud[txn.account_id] += 1

    touched = [a for a, n in per_account_fraud.items() if n > 0]
    clean = [a for a in per_account_total if per_account_fraud[a] == 0]
    if not touched or not clean:
        rep.warn("velocity_check_skipped", "not enough accounts of each kind to compare")
        return

    touched_counts = np.array([per_account_total[a] for a in touched], dtype=float)
    clean_counts = np.array([per_account_total[a] for a in clean], dtype=float)
    rep.measure(
        "account_activity",
        {
            "attacked_median": float(np.median(touched_counts)),
            "clean_median": float(np.median(clean_counts)),
            "clean_p99": float(np.percentile(clean_counts, 99)),
            "attacked_max": float(touched_counts.max()),
        },
    )
    # If the busiest 1% of untouched accounts are quieter than the typical attacked
    # account, raw volume would be close to a giveaway.
    if float(np.percentile(clean_counts, 99)) < float(np.median(touched_counts)):
        rep.error(
            "velocity_separable",
            "attacked accounts are busier than the 99th percentile of clean accounts; "
            "per-account volume alone would identify them",
        )

    overlap = _overlap_coefficient(touched_counts, clean_counts)
    rep.measure("account_activity_overlap", round(overlap, 4))
    if overlap < 0.3:
        rep.warn(
            "velocity_overlap_low",
            f"account activity distributions overlap only {overlap:.2f}",
        )


def _check_benign_sharing(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """"Shared device ⇒ fraud" must be false.

    The ecosystem seeds households that share a device and NAT/CGNAT IPs shared by
    many accounts. If every multi-account device belonged to a ring, device sharing
    would be a perfect rule and the coordinated-abuse task would be trivial.
    """
    device_accounts: Dict[str, set] = defaultdict(set)
    device_fraud: Dict[str, int] = defaultdict(int)
    ip_accounts: Dict[str, set] = defaultdict(set)
    for txn in ds.transactions:
        device_accounts[txn.device_id].add(txn.account_id)
        ip_accounts[txn.ip_id].add(txn.account_id)
        if txn.is_fraud:
            device_fraud[txn.device_id] += 1

    shared_devices = [d for d, accts in device_accounts.items() if len(accts) > 1]
    benign_shared = [d for d in shared_devices if device_fraud[d] == 0]
    shared_ips = [i for i, accts in ip_accounts.items() if len(accts) > 1]

    rep.measure("devices_used_by_multiple_accounts", len(shared_devices))
    rep.measure("benign_shared_devices", len(benign_shared))
    rep.measure("ips_used_by_multiple_accounts", len(shared_ips))
    rep.measure(
        "flagged_legitimate_shared_devices",
        sum(1 for d in ds.ecosystem.devices if d.is_shared_legitimate),
    )

    if not shared_devices:
        rep.error("no_device_sharing", "no device is used by more than one account")
    elif not benign_shared:
        rep.error(
            "device_sharing_implies_fraud",
            "every multi-account device carries fraud; device sharing alone would "
            "be a perfect detector",
        )
    else:
        share = len(benign_shared) / len(shared_devices)
        rep.measure("benign_share_of_shared_devices", round(share, 4))
        if share < 0.2:
            rep.warn(
                "benign_sharing_rare",
                f"only {share:.1%} of multi-account devices are fraud-free",
            )
    if not shared_ips:
        rep.warn("no_ip_sharing", "no IP is used by more than one account")


def _check_ecosystem_composition(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Attack-created entities must not come to dominate their populations.

    See :func:`simulator.attacks.base.capped_entities`. Also reports entities that
    appear *only* in fraudulent rows: such nodes are an unavoidable consequence of
    fabricating infrastructure, but a large share is a shortcut later checkpoints
    must not lean on, so the figure is surfaced rather than hidden.
    """
    eco = ds.ecosystem
    fake = sum(1 for m in eco.merchants if m.is_fake)
    synthetic = sum(1 for c in eco.customers if c.is_synthetic_identity)
    rep.measure("merchants_total", len(eco.merchants))
    rep.measure("merchants_fake", fake)
    rep.measure("fake_merchant_share", round(fake / max(1, len(eco.merchants)), 4))
    rep.measure("customers_synthetic", synthetic)
    rep.measure(
        "synthetic_customer_share", round(synthetic / max(1, len(eco.customers)), 4)
    )
    baseline_beneficiaries = eco.config.ecosystem.n_beneficiaries
    rep.measure("beneficiaries_total", len(eco.beneficiaries))
    rep.measure("beneficiaries_baseline", baseline_beneficiaries)

    if fake / max(1, len(eco.merchants)) > 0.5:
        rep.error(
            "fake_merchants_dominate",
            f"{fake}/{len(eco.merchants)} merchants are fake; the population is no "
            "longer mostly legitimate",
        )
    if synthetic / max(1, len(eco.customers)) > 0.35:
        rep.error(
            "synthetic_customers_dominate",
            f"{synthetic}/{len(eco.customers)} customers are synthetic identities",
        )

    normal_entities: Dict[str, set] = {"device": set(), "ip": set(), "merchant": set(),
                                       "beneficiary": set(), "account": set()}
    fraud_entities: Dict[str, set] = {k: set() for k in normal_entities}
    for txn in ds.transactions:
        bucket = fraud_entities if txn.is_fraud else normal_entities
        bucket["device"].add(txn.device_id)
        bucket["ip"].add(txn.ip_id)
        bucket["account"].add(txn.account_id)
        if txn.merchant_id:
            bucket["merchant"].add(txn.merchant_id)
        if txn.beneficiary_id:
            bucket["beneficiary"].add(txn.beneficiary_id)

    fraud_only = {
        kind: len(fraud_entities[kind] - normal_entities[kind]) for kind in fraud_entities
    }
    rep.measure("entities_appearing_only_in_fraud", fraud_only)
    rep.measure(
        "attacked_accounts_also_in_normal_traffic",
        len(fraud_entities["account"] & normal_entities["account"]),
    )

    # Embedding requirement: recruited accounts must still carry legitimate traffic,
    # or rings would be isolated components (docs/ML_READINESS.md §8).
    if fraud_entities["account"]:
        embedded = len(fraud_entities["account"] & normal_entities["account"]) / len(
            fraud_entities["account"]
        )
        rep.measure("attacked_account_embedding_rate", round(embedded, 4))
        if embedded < 0.75:
            rep.error(
                "rings_not_embedded",
                f"only {embedded:.1%} of attacked accounts also appear in legitimate "
                "traffic; rings are too isolated",
            )


def _check_seasonality(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Normal traffic must show hour-of-day and day-of-week structure."""
    hours = np.zeros(24, dtype=float)
    dows = np.zeros(7, dtype=float)
    for txn in ds.transactions:
        if txn.is_fraud:
            continue
        hours[txn.timestamp.hour] += 1
        dows[txn.timestamp.weekday()] += 1
    if hours.sum() == 0:  # pragma: no cover - defensive
        rep.error("no_normal_traffic", "dataset contains no legitimate transactions")
        return
    hp = hours / hours.sum()
    dp = dows / dows.sum()
    rep.measure("hour_share_min_max", [round(float(hp.min()), 4), round(float(hp.max()), 4)])
    rep.measure("dow_share_min_max", [round(float(dp.min()), 4), round(float(dp.max()), 4)])
    # Uniform would be 1/24 and 1/7; require visible deviation from flat.
    if float(hp.max() / hp.min()) < 1.5:
        rep.error("no_hourly_seasonality", "hour-of-day distribution is nearly uniform")
    if float(dp.max() / dp.min()) < 1.1:
        rep.warn("weak_weekly_seasonality", "day-of-week distribution is nearly uniform")


def _check_attack_parameterization(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Attacks of the same family must vary, and differ across splits.

    Identical footprints for every instance would mean the family has a hardcoded
    shape (PRD FR-2), and identical parameters across splits would defeat the
    distribution shift the split protocol calls for.
    """
    per_family_split: Dict[tuple, List[int]] = defaultdict(list)
    for record in ds.ground_truth:
        per_family_split[(record.attack_type, record.split)].append(
            len(record.affected_transaction_ids)
        )

    identical: List[str] = []
    for (family, split), sizes in sorted(per_family_split.items()):
        if len(sizes) > 2 and len(set(sizes)) == 1:
            identical.append(f"{family}/{split}")
    rep.measure("families_with_constant_footprint", identical)

    # Parameters must actually differ between splits.
    params_by_split: Dict[str, Dict[str, Any]] = {}
    for record in ds.ground_truth:
        params_by_split.setdefault(record.split, {}).setdefault(
            record.attack_type, record.generation_parameters
        )
    unchanged: List[str] = []
    for family in ATTACK_FAMILIES:
        seen = [
            params_by_split.get(s, {}).get(family)
            for s in SPLIT_NAMES
            if params_by_split.get(s, {}).get(family) is not None
        ]
        if len(seen) > 1 and all(p == seen[0] for p in seen):
            unchanged.append(family)
    rep.measure("families_with_identical_params_across_splits", unchanged)
    if unchanged:
        rep.error(
            "no_distribution_shift",
            f"these families use identical parameters in every split: {unchanged}",
        )


def validate_dataset(ds: GeneratedDataset) -> ValidationReport:
    """Run every check and return a report.

    Ordered cheapest-first only incidentally; all checks always run, because a
    validation run that stops at the first error tells you less than one that
    reports everything wrong at once.
    """
    rep = ValidationReport()
    _check_ids_and_fields(ds, rep)
    _check_referential_integrity(ds, rep)
    _check_temporal_split(ds, rep)
    _check_temporal_causality(ds, rep)
    _check_prevalence(ds, rep)
    _check_ground_truth(ds, rep)
    _check_amount_non_separability(ds, rep)
    _check_velocity_non_separability(ds, rep)
    _check_benign_sharing(ds, rep)
    _check_ecosystem_composition(ds, rep)
    _check_seasonality(ds, rep)
    _check_attack_parameterization(ds, rep)
    return rep
