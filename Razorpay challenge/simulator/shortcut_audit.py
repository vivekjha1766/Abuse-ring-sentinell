"""Adversarial shortcut audit — can any single field give fraud away?

A synthetic fraud dataset is only worth building if the fraud is *hard*. The
failure mode is subtle: a generator can produce data that looks rich while
quietly encoding the label in some incidental field, so a detector scores well
for reasons that would never transfer to real data. This module attacks the
dataset on purpose, looking for exactly that.

Method
------
Each candidate shortcut is scored **standalone** — one field, no model, no
combination — against two kinds of target: "is this row fraudulent?" and "does
this row belong to family *F*?". The per-family question matters because an
aggregate can look safe while one family sits alone in an obvious corner of the
space. That is not hypothetical: it is how the microtransaction family was caught
being isolable by ``amount < 6`` at 100% recall.

Three measures are used, chosen so the verdict does not move with prevalence:

``AUC``
    Rank-based, prevalence-independent, for ordered fields. 0.5 is useless, 1.0
    is a perfect oracle. This is the primary gate.
``best band F1``
    Best F1 over any interval ``lo <= x <= hi``, chosen *with* the labels. An
    upper bound on what a threshold rule could achieve.
``precision / lift`` for boolean rules
    Lift is precision divided by the base rate, so "3.1x" means the same thing at
    1% and at 25% prevalence. Raw F1 does not: predicting everything positive
    scores F1 0.10 at 1% prevalence and 0.40 at 25%, so a raw-F1 gate would be
    far stricter at low prevalence purely as an artefact.

Identifier fields (merchant, device, IP) get the treatment a real modeller would
give them: a fraud-rate lookup is fitted **on TRAIN only** and applied to
HELDOUT. Scoring an identifier against labels it was fitted on would measure
memorisation, not leakage. Separately, each identifier is tested for the
*novelty* shortcut — "this entity was never seen in train" — which is the
characteristic weakness of any simulator that fabricates infrastructure for its
attacks.

What counts as a failure
------------------------
``AUC >= 0.98`` or a boolean rule with ``precision >= 0.95`` at ``recall >= 0.50``
is treated as an oracle and fails the audit. The correct response to a failure is
to fix the *data-generating process* — never to drop the feature or weaken the
model, which would hide the flaw rather than remove it.

This module reads labels, and like :mod:`simulator.validation` it is evaluation
infrastructure that feature and model code must never import
(``docs/FEATURE_LEAKAGE_POLICY.md`` §4).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from simulator.base_generator import haversine_km
from simulator.ground_truth import ATTACK_FAMILIES
from simulator.pipeline import WARMUP_SPLIT, GeneratedDataset

__all__ = [
    "ShortcutResult",
    "ShortcutAuditReport",
    "audit_shortcuts",
    "ORACLE_AUC",
    "STRONG_AUC",
    "ORACLE_PRECISION",
    "ORACLE_RECALL",
]

#: AUC at or above this is treated as a near-perfect oracle and fails the audit.
ORACLE_AUC = 0.98
#: AUC at or above this is reported as a strong signal, but not a failure. Some
#: fields *should* be informative — fraud that changed nothing observable would be
#: undetectable by construction. The audit objects to sufficiency, not to signal.
STRONG_AUC = 0.90
#: A boolean rule this precise, with at least ``ORACLE_RECALL`` coverage, is an oracle.
ORACLE_PRECISION = 0.95
ORACLE_RECALL = 0.50


@dataclass(frozen=True)
class ShortcutResult:
    feature: str
    target: str
    kind: str
    auc: Optional[float]
    best_f1: float
    precision: float
    recall: float
    lift: Optional[float]
    rule: str
    verdict: str

    @property
    def is_oracle(self) -> bool:
        return self.verdict == "oracle"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "target": self.target,
            "kind": self.kind,
            "auc": None if self.auc is None else round(self.auc, 4),
            "best_f1": round(self.best_f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "lift": None if self.lift is None else round(self.lift, 2),
            "rule": self.rule,
            "verdict": self.verdict,
        }


@dataclass
class ShortcutAuditReport:
    results: List[ShortcutResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: ShortcutResult) -> None:
        self.results.append(result)

    @property
    def oracles(self) -> List[ShortcutResult]:
        return [r for r in self.results if r.verdict == "oracle"]

    @property
    def strong(self) -> List[ShortcutResult]:
        return [r for r in self.results if r.verdict == "strong"]

    @property
    def ok(self) -> bool:
        return not self.oracles

    def summary(self) -> str:
        return (
            f"{'PASS' if self.ok else 'FAIL'} — {len(self.results)} shortcut tests, "
            f"{len(self.oracles)} oracle(s), {len(self.strong)} strong signal(s)"
        )

    def worst(self, n: int = 12) -> List[ShortcutResult]:
        """Highest-signal results first, for reporting."""
        return sorted(
            self.results,
            key=lambda r: (r.auc if r.auc is not None else 0.5, r.best_f1),
            reverse=True,
        )[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "oracles": [r.to_dict() for r in self.oracles],
            "strong": [r.to_dict() for r in self.strong],
            "all_results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
            "measurements": self.measurements,
        }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney U), tie-corrected.

    Prevalence-independent, which is why it is the primary gate: the same
    threshold is meaningful at 1% and at 25% fraud.

    Returned as ``max(auc, 1 - auc)``. A field that predicts fraud perfectly when
    *inverted* is exactly as dangerous as one that predicts it directly, and a
    detector would find either.
    """
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = float(ranks[labels].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(auc, 1.0 - auc)


def _best_band(
    values: np.ndarray, labels: np.ndarray, *, grid: int = 60
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Best F1 over any interval of ``values``, with its precision and recall.

    Sweeps every pair of grid quantiles on the sorted array using a prefix sum, so
    each candidate band is O(1). Uses the labels deliberately — this is an upper
    bound on threshold-rule power, not an honest estimate of generalization.
    """
    n_pos = int(labels.sum())
    if values.size == 0 or n_pos == 0:
        return 0.0, 0.0, 0.0, (0.0, 0.0)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_labels = labels[order].astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(sorted_labels)])
    cuts = np.unique(np.linspace(0, values.size, grid + 1).astype(int))
    best = (0.0, 0.0, 0.0, (0.0, 0.0))
    for a in range(cuts.size):
        lo_i = int(cuts[a])
        for b in range(a + 1, cuts.size):
            hi_i = int(cuts[b])
            tp = int(prefix[hi_i] - prefix[lo_i])
            if tp == 0:
                continue
            precision = tp / (hi_i - lo_i)
            recall = tp / n_pos
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best[0]:
                best = (
                    f1,
                    precision,
                    recall,
                    (float(sorted_values[lo_i]), float(sorted_values[hi_i - 1])),
                )
    return best


def _verdict_numeric(auc: Optional[float]) -> str:
    if auc is None:
        return "skipped"
    if auc >= ORACLE_AUC:
        return "oracle"
    if auc >= STRONG_AUC:
        return "strong"
    return "ok"


def _verdict_boolean(precision: float, recall: float, lift: Optional[float]) -> str:
    if precision >= ORACLE_PRECISION and recall >= ORACLE_RECALL:
        return "oracle"
    if lift is not None and lift >= 5.0 and recall >= 0.25:
        return "strong"
    return "ok"


# --------------------------------------------------------------------------
# Target construction
# --------------------------------------------------------------------------


def _targets(ds: GeneratedDataset, mask: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """"Any fraud" plus one target per attack family, restricted to ``mask`` rows."""
    family_of: Dict[str, str] = {}
    for record in ds.ground_truth:
        for tid in record.affected_transaction_ids:
            family_of[tid] = record.attack_type
    rows = [t for t, keep in zip(ds.transactions, mask) if keep]
    fraud = np.array([t.is_fraud for t in rows], dtype=bool)
    out: List[Tuple[str, np.ndarray]] = [("any_fraud", fraud)]
    families = np.array([family_of.get(t.transaction_id, "") for t in rows])
    for family in ATTACK_FAMILIES:
        target = families == family
        if target.sum() >= 25:
            out.append((family, target))
    return out


def _score_numeric(
    rep: ShortcutAuditReport,
    feature: str,
    values: np.ndarray,
    targets: Sequence[Tuple[str, np.ndarray]],
    *,
    kind: str = "numeric",
) -> None:
    finite = np.isfinite(values)
    for name, target in targets:
        usable = finite & np.isfinite(values)
        if usable.sum() < 100 or target[usable].sum() < 10:
            continue
        v = values[usable]
        t = target[usable]
        auc = _auc(v, t)
        f1, precision, recall, band = _best_band(v, t)
        rep.add(
            ShortcutResult(
                feature=feature,
                target=name,
                kind=kind,
                auc=auc,
                best_f1=f1,
                precision=precision,
                recall=recall,
                lift=None,
                rule=f"{band[0]:.4g} <= x <= {band[1]:.4g}",
                verdict=_verdict_numeric(auc),
            )
        )


def _score_boolean(
    rep: ShortcutAuditReport,
    feature: str,
    flags: np.ndarray,
    targets: Sequence[Tuple[str, np.ndarray]],
    *,
    rule: str,
    kind: str = "boolean",
) -> None:
    for name, target in targets:
        n_pos = int(target.sum())
        predicted = int(flags.sum())
        if n_pos == 0:
            continue
        tp = int((flags & target).sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / n_pos
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        base = n_pos / target.size
        lift = (precision / base) if base > 0 else None
        rep.add(
            ShortcutResult(
                feature=feature,
                target=name,
                kind=kind,
                auc=_auc(flags.astype(np.float64), target),
                best_f1=f1,
                precision=precision,
                recall=recall,
                lift=lift,
                rule=rule,
                verdict=_verdict_boolean(precision, recall, lift),
            )
        )


# --------------------------------------------------------------------------
# Individual shortcut probes
# --------------------------------------------------------------------------


def _probe_amount(ds, rows, targets, rep) -> None:
    _score_numeric(rep, "amount", np.array([t.amount for t in rows]), targets)
    # Amount expressed against the account's own baseline. A detector would build
    # this ratio immediately, so it must be checked as its own shortcut.
    accounts = ds.ecosystem.account_by_id
    ratio = np.array(
        [
            t.amount / max(1.0, accounts[t.account_id].baseline_amount_mean)
            if t.account_id in accounts
            else np.nan
            for t in rows
        ]
    )
    _score_numeric(rep, "amount_over_account_baseline", ratio, targets)
    # Cents pattern: round-number amounts would betray hand-built attack values.
    cents = np.array([round((t.amount % 1.0) * 100) for t in rows], dtype=float)
    _score_numeric(rep, "amount_cents", cents, targets)
    is_round = np.array([abs(t.amount - round(t.amount)) < 1e-9 for t in rows])
    _score_boolean(rep, "amount_is_whole_number", is_round, targets, rule="amount % 1 == 0")


def _probe_velocity(ds, rows, targets, rep) -> None:
    total: Dict[str, int] = defaultdict(int)
    for txn in rows:
        total[txn.account_id] += 1
    _score_numeric(
        rep,
        "account_transaction_count",
        np.array([float(total[t.account_id]) for t in rows]),
        targets,
    )

    # Local burst rate: how many of this account's transactions fall within one
    # hour of this one. This is the feature a velocity rule would actually use, and
    # it is far sharper than a whole-period count.
    by_account: Dict[str, List[float]] = defaultdict(list)
    for txn in rows:
        by_account[txn.account_id].append(txn.timestamp.timestamp())
    for key in by_account:
        by_account[key].sort()
    burst = np.empty(len(rows), dtype=float)
    gap = np.empty(len(rows), dtype=float)
    for i, txn in enumerate(rows):
        times = by_account[txn.account_id]
        ts = txn.timestamp.timestamp()
        lo = np.searchsorted(times, ts - 3600.0, side="left")
        hi = np.searchsorted(times, ts + 3600.0, side="right")
        burst[i] = hi - lo
        pos = np.searchsorted(times, ts, side="left")
        gap[i] = ts - times[pos - 1] if pos > 0 else np.nan
    _score_numeric(rep, "account_burst_count_1h", burst, targets)
    _score_numeric(rep, "seconds_since_account_previous_txn", gap, targets)


def _probe_identity(ds, rows, targets, rep, split_of) -> None:
    """Merchant / device / IP identity, fitted on TRAIN and applied to HELDOUT.

    Two distinct shortcuts are separated here. A *fraud-rate lookup* asks whether
    knowing which entity was involved predicts fraud — legitimate signal up to a
    point, since real rings do reuse infrastructure. A *novelty* rule asks whether
    "never seen in train" predicts fraud, which is the artefact any simulator that
    fabricates attack infrastructure is prone to.
    """
    train_rows = [t for t in ds.transactions if split_of[t.transaction_id] == "train"]
    for label, getter in (
        ("merchant", lambda t: t.merchant_id),
        ("device", lambda t: t.device_id),
        ("ip", lambda t: t.ip_id),
        ("beneficiary", lambda t: t.beneficiary_id),
        ("account", lambda t: t.account_id),
    ):
        fraud_count: Dict[str, int] = defaultdict(int)
        seen_count: Dict[str, int] = defaultdict(int)
        for txn in train_rows:
            key = getter(txn)
            if key is None:
                continue
            seen_count[key] += 1
            if txn.is_fraud:
                fraud_count[key] += 1

        # Smoothed fraud rate learned on train; unseen entities get the prior.
        prior = sum(fraud_count.values()) / max(1, sum(seen_count.values()))
        rate = np.array(
            [
                (fraud_count[k] + 5.0 * prior) / (seen_count[k] + 5.0)
                if (k := getter(t)) is not None and k in seen_count
                else prior
                for t in rows
            ]
        )
        _score_numeric(
            rep, f"{label}_train_fraud_rate", rate, targets, kind="identifier"
        )

        unseen = np.array(
            [
                (k := getter(t)) is not None and k not in seen_count
                for t in rows
            ]
        )
        if unseen.sum() >= 10:
            _score_boolean(
                rep,
                f"{label}_unseen_in_train",
                unseen,
                targets,
                rule=f"{label} never appears in TRAIN",
                kind="novelty",
            )


def _probe_account_age(ds, rows, targets, rep) -> None:
    accounts = ds.ecosystem.account_by_id
    customers = ds.ecosystem.customer_by_id
    acc_age = np.empty(len(rows), dtype=float)
    cust_age = np.empty(len(rows), dtype=float)
    freq = np.empty(len(rows), dtype=float)
    for i, txn in enumerate(rows):
        account = accounts.get(txn.account_id)
        if account is None:
            acc_age[i] = cust_age[i] = freq[i] = np.nan
            continue
        acc_age[i] = (txn.timestamp - account.created_at).total_seconds() / 86400.0
        freq[i] = account.baseline_tx_frequency
        customer = customers.get(account.customer_id)
        cust_age[i] = (
            (txn.timestamp - customer.created_at).total_seconds() / 86400.0
            if customer
            else np.nan
        )
    _score_numeric(rep, "account_age_days", acc_age, targets)
    _score_numeric(rep, "customer_age_days", cust_age, targets)
    _score_numeric(rep, "account_baseline_tx_frequency", freq, targets)


def _probe_geography(ds, rows, targets, rep) -> None:
    customers = ds.ecosystem.customer_by_id
    accounts = ds.ecosystem.account_by_id
    dist = np.empty(len(rows), dtype=float)
    for i, txn in enumerate(rows):
        account = accounts.get(txn.account_id)
        customer = customers.get(account.customer_id) if account else None
        dist[i] = (
            haversine_km(customer.home_lat, customer.home_lon, txn.lat, txn.lon)
            if customer
            else np.nan
        )
    _score_numeric(rep, "distance_from_home_km", dist, targets)
    _score_numeric(rep, "latitude", np.array([t.lat for t in rows]), targets)
    _score_numeric(rep, "longitude", np.array([t.lon for t in rows]), targets)


def _probe_entity_creation(ds, rows, targets, rep) -> None:
    """Creation timestamps of the entities a row touches.

    The suspicion this probe exists to test: attack families create devices, IPs
    and merchants shortly before using them, so "this device was registered hours
    ago" can become a giveaway that has nothing to do with fraudulent *behaviour*.
    """
    eco = ds.ecosystem
    devices = eco.device_by_id
    ips = eco.ip_by_id
    merchants = eco.merchant_by_id
    beneficiaries = eco.beneficiary_by_id

    def age_days(created: Optional[datetime], ts: datetime) -> float:
        return np.nan if created is None else (ts - created).total_seconds() / 86400.0

    for label, table, getter in (
        ("device", devices, lambda t: t.device_id),
        ("ip", ips, lambda t: t.ip_id),
        ("merchant", merchants, lambda t: t.merchant_id),
        ("beneficiary", beneficiaries, lambda t: t.beneficiary_id),
    ):
        ages = np.empty(len(rows), dtype=float)
        for i, txn in enumerate(rows):
            key = getter(txn)
            entity = table.get(key) if key is not None else None
            ages[i] = age_days(getattr(entity, "created_at", None), txn.timestamp)
        _score_numeric(rep, f"{label}_age_days_at_txn", ages, targets, kind="creation_time")


def _probe_sharing_structure(ds, rows, targets, rep) -> None:
    """How many *other* entities a row's entities touch, fitted on TRAIN.

    The gap this closes. ``_probe_identity`` asks whether knowing *which* device was
    involved predicts fraud; it does not ask whether knowing *how many accounts that
    device serves* predicts fraud. Sharing degree is the single most natural feature
    a coordinated-abuse model would reach for, and it is the one a simulator is most
    likely to make trivial: ``shared_device_ring`` and ``synthetic_identity_cluster``
    both work by pointing many accounts at one device, so unless the legitimate
    population shares infrastructure at comparable intensity, "degree > k" becomes a
    rule that finds Class B abuse without modelling anything.

    ``simulator.validation._check_benign_sharing`` warns when few multi-account
    devices are entirely fraud-free, which at high prevalence is nearly guaranteed by
    the base rate alone and says little about separability. This probe measures the
    thing that actually matters — the standalone ranking power of the degree itself.

    Degrees are counted on TRAIN rows only and applied to every scored row, matching
    the identifier probes: a model deployed at the boundary knows the history it has
    seen, not the future.
    """
    split_of = ds.split_of
    train_rows = [t for t in ds.transactions if split_of[t.transaction_id] == "train"]

    # (label, key on the row, the entity whose distinct count we accumulate)
    pairs = (
        ("device_account_degree", lambda t: t.device_id, lambda t: t.account_id),
        ("ip_account_degree", lambda t: t.ip_id, lambda t: t.account_id),
        ("merchant_account_degree", lambda t: t.merchant_id, lambda t: t.account_id),
        ("beneficiary_account_degree", lambda t: t.beneficiary_id, lambda t: t.account_id),
        ("account_device_degree", lambda t: t.account_id, lambda t: t.device_id),
        ("account_ip_degree", lambda t: t.account_id, lambda t: t.ip_id),
        ("account_merchant_degree", lambda t: t.account_id, lambda t: t.merchant_id),
        (
            "account_beneficiary_degree",
            lambda t: t.account_id,
            lambda t: t.beneficiary_id,
        ),
    )
    for label, key_of, other_of in pairs:
        degree: Dict[str, set] = defaultdict(set)
        for txn in train_rows:
            key = key_of(txn)
            other = other_of(txn)
            if key is None or other is None:
                continue
            degree[key].add(other)
        sizes = {k: float(len(v)) for k, v in degree.items()}
        values = np.array(
            [
                sizes.get(k, np.nan) if (k := key_of(t)) is not None else np.nan
                for t in rows
            ]
        )
        _score_numeric(rep, label, values, targets, kind="sharing_structure")


def _probe_merchant_profile(ds, rows, targets, rep) -> None:
    """A merchant's own declared attributes, independent of who transacted with it.

    Shell merchants are drawn from the legitimate *small* tier deliberately — see
    :func:`simulator.entities._generate_merchants` — but "deliberately" is an
    intention, not a measurement. If shells are the only small-tier merchants that
    carry meaningful volume, or if their ticket size sits apart from the legitimate
    small-tier spread, then ``size_tier`` and ``avg_ticket_size`` leak ring
    membership without any behavioural modelling. Both are legitimate model inputs,
    so neither may be an oracle.
    """
    merchants = ds.ecosystem.merchant_by_id
    tiers = sorted(
        {m.size_tier for m in ds.ecosystem.merchants if m.size_tier is not None}
    )
    for tier in tiers:
        flags = np.array(
            [
                getattr(merchants.get(t.merchant_id), "size_tier", None) == tier
                if t.merchant_id is not None
                else False
                for t in rows
            ]
        )
        _score_boolean(
            rep,
            f"merchant_size_tier=={tier}",
            flags,
            targets,
            rule=f"merchant.size_tier == {tier!r}",
            kind="categorical",
        )

    ticket = np.array(
        [
            float(getattr(merchants.get(t.merchant_id), "avg_ticket_size", np.nan))
            if t.merchant_id is not None
            else np.nan
            for t in rows
        ],
        dtype=float,
    )
    _score_numeric(rep, "merchant_avg_ticket_size", ticket, targets)
    # The row's own amount relative to what the merchant usually charges. A shell
    # whose ring purchases ignore its declared ticket size would show up here even
    # though neither field alone looked wrong.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.array([t.amount for t in rows], dtype=float) / ticket
    _score_numeric(rep, "amount_over_merchant_ticket", ratio, targets)


def _probe_generator_metadata(ds, rows, targets, rep) -> None:
    """Fields that carry no business meaning and must therefore carry no signal.

    Identifier text, the entity's position in creation order, and the low-cardinality
    enums. Creation order is the sharpest of these: entities appended by attacks sit
    at the end of every list, so *index* can leak attack membership even though the
    hashed ID string does not.
    """
    eco = ds.ecosystem
    index_of: Dict[str, int] = {}
    for seq, attr in (
        (eco.devices, "device_id"),
        (eco.ips, "ip_id"),
        (eco.merchants, "merchant_id"),
        (eco.beneficiaries, "beneficiary_id"),
        (eco.accounts, "account_id"),
        (eco.customers, "customer_id"),
    ):
        for i, entity in enumerate(seq):
            index_of[getattr(entity, attr)] = i

    for label, getter in (
        ("device", lambda t: t.device_id),
        ("ip", lambda t: t.ip_id),
        ("merchant", lambda t: t.merchant_id),
        ("beneficiary", lambda t: t.beneficiary_id),
        ("account", lambda t: t.account_id),
    ):
        idx = np.array(
            [
                float(index_of.get(k, np.nan)) if (k := getter(t)) is not None else np.nan
                for t in rows
            ]
        )
        _score_numeric(
            rep, f"{label}_creation_order_index", idx, targets, kind="generator_metadata"
        )

    # The transaction ID itself: its lexical content and its position in the
    # emission sequence. Both must be uninformative.
    tid_numeric = np.array(
        [float(int(t.transaction_id.split("_")[1][:6], 36)) for t in rows]
    )
    _score_numeric(
        rep, "transaction_id_lexical_value", tid_numeric, targets, kind="generator_metadata"
    )
    _score_numeric(
        rep,
        "transaction_row_order",
        np.arange(len(rows), dtype=float),
        targets,
        kind="generator_metadata",
    )

    for label, values in (
        ("channel", [t.channel for t in rows]),
        ("status", [t.status for t in rows]),
        ("transaction_type", [t.transaction_type for t in rows]),
        ("currency", [t.currency for t in rows]),
    ):
        for level in sorted(set(values)):
            flags = np.array([v == level for v in values])
            if flags.sum() < 50:
                continue
            _score_boolean(
                rep,
                f"{label}=={level}",
                flags,
                targets,
                rule=f"{label} == {level!r}",
                kind="categorical",
            )


def _probe_attack_specific_fields(ds, rows, targets, rep) -> None:
    """Entity flags that exist for ground truth and must stay quarantined.

    ``is_fake``, ``is_synthetic_identity`` and ``is_shared_legitimate`` are
    generator bookkeeping. They are expected to be strong — that is precisely why
    they are quarantined — so their power is *measured and recorded* here to
    justify the quarantine, rather than being left as an unexamined assumption.
    Results are tagged ``quarantined_field`` and do not fail the audit; the
    enforcement that they never reach a model lives in
    :func:`simulator.leakage.assert_feature_safe`.
    """
    eco = ds.ecosystem
    merchants = eco.merchant_by_id
    customers = eco.customer_by_id
    devices = eco.device_by_id

    probes = (
        (
            "merchant.is_fake",
            np.array(
                [
                    bool(getattr(merchants.get(t.merchant_id), "is_fake", False))
                    for t in rows
                ]
            ),
        ),
        (
            "customer.is_synthetic_identity",
            np.array(
                [
                    bool(
                        getattr(
                            customers.get(t.customer_id), "is_synthetic_identity", False
                        )
                    )
                    for t in rows
                ]
            ),
        ),
        (
            "device.is_shared_legitimate",
            np.array(
                [
                    bool(
                        getattr(devices.get(t.device_id), "is_shared_legitimate", False)
                    )
                    for t in rows
                ]
            ),
        ),
    )
    for name, flags in probes:
        if flags.sum() == 0:
            continue
        for target_name, target in targets:
            n_pos = int(target.sum())
            predicted = int(flags.sum())
            tp = int((flags & target).sum())
            precision = tp / predicted if predicted else 0.0
            recall = tp / n_pos if n_pos else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            base = n_pos / target.size
            rep.add(
                ShortcutResult(
                    feature=name,
                    target=target_name,
                    kind="quarantined_field",
                    auc=_auc(flags.astype(np.float64), target),
                    best_f1=f1,
                    precision=precision,
                    recall=recall,
                    lift=(precision / base) if base > 0 else None,
                    rule=f"{name} is True",
                    # Never "oracle": these are ground truth by design, and the
                    # audit's job here is to quantify them, not to fail on them.
                    verdict="quarantined",
                )
            )


def audit_shortcuts(ds: GeneratedDataset, *, verbose: bool = False) -> ShortcutAuditReport:
    """Run every shortcut probe over the evaluation window.

    Warm-up rows are excluded: they are normal-only by construction, so including
    them would inflate every measure with rows no detector is ever scored on.
    """
    rep = ShortcutAuditReport()
    mask = np.array(
        [ds.split_of[t.transaction_id] != WARMUP_SPLIT for t in ds.transactions]
    )
    rows = [t for t, keep in zip(ds.transactions, mask) if keep]
    targets = _targets(ds, mask)
    rep.measurements["rows_audited"] = len(rows)
    rep.measurements["targets"] = {name: int(t.sum()) for name, t in targets}

    probes: Tuple[Tuple[str, Callable], ...] = (
        ("amount", _probe_amount),
        ("velocity", _probe_velocity),
        ("account_age", _probe_account_age),
        ("geography", _probe_geography),
        ("entity_creation_time", _probe_entity_creation),
        ("sharing_structure", _probe_sharing_structure),
        ("merchant_profile", _probe_merchant_profile),
        ("generator_metadata", _probe_generator_metadata),
        ("attack_specific_fields", _probe_attack_specific_fields),
    )
    for name, probe in probes:
        if verbose:
            print(f"  probing {name} ...", flush=True)
        probe(ds, rows, targets, rep)
    if verbose:
        print("  probing identity ...", flush=True)
    _probe_identity(ds, rows, targets, rep, ds.split_of)

    rep.measurements["tests_run"] = len(rep.results)
    rep.measurements["oracle_count"] = len(rep.oracles)
    rep.measurements["strong_count"] = len(rep.strong)
    return rep
