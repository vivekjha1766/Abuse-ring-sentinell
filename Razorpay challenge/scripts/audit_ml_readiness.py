#!/usr/bin/env python
"""The Checkpoint 1.5 ML-readiness gate.

``docs/ML_READINESS.md`` lists every criterion the dataset must satisfy before any
feature or model work begins. This script measures them and returns a non-zero exit
code if the gate fails.

**It composes, it does not duplicate.** Most of the checklist is already measured by
Checkpoint 1 code, so this script runs :func:`simulator.validation.validate_dataset`
and :func:`simulator.shortcut_audit.audit_shortcuts` and reports their verdicts
under the section each belongs to. It adds only the checks nothing covers yet:

===============================  ===========================================
``ML_READINESS`` reference       added here
===============================  ===========================================
§2 duplicate transactions        exact and near-duplicate row detection
§4 both classes per split        an explicit single-class failure
§5 fraud not time-confined       per-day fraud coverage and the longest gap
§5 warm-up sufficiency           warm-up span against the longest window
§6 ground-truth quarantine       an AST scan of the non-simulator packages
§6/§7 point-in-time at scale     recompute-after-truncation on real rows
§8 graph readiness               connectivity, degree, rings, cycles
===============================  ===========================================

**Not a second test framework.** It reuses
:class:`simulator.validation.ValidationReport` for issues and measurements rather
than inventing a parallel one, has no discovery, no fixtures, and no assertions —
one function per checklist section, each recording what it measured. The unit-level
guards live in ``tests/leakage/``; this is the at-scale confirmation on a real
dataset.

**On importing ground truth.** This script does import
:mod:`simulator.ground_truth`, which is exactly what ``ML_READINESS`` §6 forbids for
*feature, model and scoring* modules. An audit is none of those: measuring whether
rings are embedded in the legitimate graph requires knowing which entities are in a
ring. The quarantine is enforced where it matters — :func:`check_ground_truth_quarantine`
scans the feature and model packages and would fail on this file's imports if it
ever moved there.

**Cost.** One dataset generation per prevalence, so one prevalence per invocation.
No model is trained and no threshold is chosen; those need scores that do not exist
until Checkpoint 2.

Usage::

    python scripts/audit_ml_readiness.py                          # seed 42, 5%
    python scripts/audit_ml_readiness.py --prevalence 0.25
    python scripts/audit_ml_readiness.py --json artifacts/audit_5.json
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/audit_ml_readiness.py`
    sys.path.insert(0, str(REPO_ROOT))

from ml.point_in_time import TransactionHistory
from simulator.config import SPLIT_NAMES, SUPPORTED_PREVALENCES, default_config
from simulator.leakage import LABEL_FIELDS
from simulator.pipeline import WARMUP_SPLIT, GeneratedDataset, generate_dataset
from simulator.shortcut_audit import audit_shortcuts
from simulator.validation import ValidationReport, validate_dataset

#: The longest feature window planned in ``docs/FEATURE_REFERENCE.md``. Warm-up must
#: cover it or the first evaluation day's rolling features are silently truncated.
LONGEST_FEATURE_WINDOW_DAYS: int = 7

#: Packages that must never reach the ground-truth registry (``ML_READINESS`` §6).
#: Only those that exist are scanned; the later checkpoints add the rest.
QUARANTINED_PACKAGES: Tuple[str, ...] = ("ml", "graph", "backend")

#: How many transactions the at-scale point-in-time check recomputes. Large enough
#: to cover every attack family and cold-start accounts, small enough to stay well
#: inside a couple of seconds.
PIT_SAMPLE_SIZE: int = 4_000

#: Fixed, so two runs of this audit sample the same rows.
PIT_SAMPLE_SEED: int = 20260824


# --------------------------------------------------------------------------
# §2 Dataset sanity — duplicates
# --------------------------------------------------------------------------


def check_duplicate_rows(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Exact and near-duplicate transactions (``ML_READINESS`` §2).

    An exact duplicate is the same row twice under two IDs — a generator that
    injected the same transaction into two splits would produce one, and a model
    would then be tested on a row it trained on. A near-duplicate shares everything
    a detector could observe (account, merchant, amount, and the minute) without
    being byte-identical; a handful are expected in real traffic, an epidemic of
    them means the amount or timing draw has collapsed.
    """
    exact = Counter()
    near = Counter()
    for t in ds.transactions:
        exact[
            (
                t.account_id,
                t.merchant_id,
                t.beneficiary_id,
                t.device_id,
                t.ip_id,
                round(t.amount, 2),
                t.timestamp,
                t.channel,
                t.transaction_type,
            )
        ] += 1
        near[
            (
                t.account_id,
                t.merchant_id,
                round(t.amount, 2),
                t.timestamp.replace(second=0, microsecond=0),
            )
        ] += 1

    exact_dupes = sum(c - 1 for c in exact.values() if c > 1)
    near_dupes = sum(c - 1 for c in near.values() if c > 1)
    total = len(ds.transactions)
    near_share = near_dupes / total if total else 0.0

    rep.measure("duplicate_rows_exact", exact_dupes)
    rep.measure("duplicate_rows_near", near_dupes)
    rep.measure("duplicate_rows_near_share", round(near_share, 5))

    if exact_dupes:
        rep.error(
            "duplicate_transactions",
            f"{exact_dupes} transactions are identical to another row in every "
            "observable field",
        )
    if near_share > 0.01:
        rep.warn(
            "near_duplicate_transactions",
            f"{near_dupes} rows ({near_share:.2%}) repeat an "
            "(account, merchant, amount, minute) tuple",
        )


# --------------------------------------------------------------------------
# §4 Class balance — both classes per split
# --------------------------------------------------------------------------


def check_both_classes_present(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """No evaluation split may be single-class (``ML_READINESS`` §4).

    ``_check_prevalence`` warns when a split's fraud rate drifts; this fails when a
    split has no fraud at all, which is a different problem: PR-AUC is undefined and
    a threshold chosen on that split is meaningless.
    """
    counts = ds.split_counts()
    for split in SPLIT_NAMES:
        c = counts[split]
        if c["fraud"] == 0:
            rep.error("single_class_split", f"{split} contains no fraudulent rows")
        elif c["normal"] == 0:
            rep.error("single_class_split", f"{split} contains no normal rows")
    rep.measure(
        "class_counts_by_split",
        {s: {"fraud": counts[s]["fraud"], "normal": counts[s]["normal"]} for s in SPLIT_NAMES},
    )


# --------------------------------------------------------------------------
# §5 Temporal distribution
# --------------------------------------------------------------------------


def check_fraud_not_time_confined(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Fraud must be woven through the timeline (``ML_READINESS`` §5).

    Two ways to be confined, and both are measured. Fraud could occupy one
    contiguous block — caught by the fraud-free-day gap. Or it could be spread thin
    but spiked, with most fraud on one day — caught by the busiest day's share. A
    dataset failing either lets a model learn the calendar instead of the behaviour.
    """
    fraud_by_day: Counter = Counter()
    all_days = set()
    for t in ds.transactions:
        if ds.split_of[t.transaction_id] == WARMUP_SPLIT:
            continue
        day = t.timestamp.date()
        all_days.add(day)
        if t.is_fraud:
            fraud_by_day[day] += 1

    if not all_days:
        rep.error("no_evaluation_rows", "the evaluation window contains no transactions")
        return

    ordered = sorted(all_days)
    days_with_fraud = sum(1 for d in ordered if fraud_by_day[d])
    coverage = days_with_fraud / len(ordered)

    longest_gap = current = 0
    for day in ordered:
        current = 0 if fraud_by_day[day] else current + 1
        longest_gap = max(longest_gap, current)

    total_fraud = sum(fraud_by_day.values())
    busiest_share = (max(fraud_by_day.values()) / total_fraud) if total_fraud else 0.0

    rep.measure("evaluation_days", len(ordered))
    rep.measure("days_with_fraud", days_with_fraud)
    rep.measure("fraud_day_coverage", round(coverage, 4))
    rep.measure("longest_fraud_free_gap_days", longest_gap)
    rep.measure("busiest_fraud_day_share", round(busiest_share, 4))

    if coverage < 0.5:
        rep.error(
            "fraud_time_confined",
            f"fraud appears on only {days_with_fraud}/{len(ordered)} evaluation days",
        )
    if longest_gap > max(7, len(ordered) // 4):
        rep.error(
            "fraud_time_confined",
            f"{longest_gap} consecutive evaluation days contain no fraud",
        )
    if busiest_share > 0.5:
        rep.error(
            "fraud_time_confined",
            f"one day holds {busiest_share:.1%} of all fraudulent rows",
        )


def check_warmup_sufficiency(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """Warm-up must cover the longest feature window (``ML_READINESS`` §5).

    Span alone is not enough: a 14-day warm-up with no rows for an account gives
    that account's 7-day features nothing to average, so the share of evaluation
    accounts with warm-up history is measured too.
    """
    warmup = [t for t in ds.transactions if ds.split_of[t.transaction_id] == WARMUP_SPLIT]
    if not warmup:
        rep.error("no_warmup", "warm-up window contains no transactions")
        return

    span_days = (max(t.timestamp for t in warmup) - min(t.timestamp for t in warmup)).days
    configured = ds.config.warmup_days
    warmup_accounts = {t.account_id for t in warmup}
    train_accounts = {
        t.account_id for t in ds.transactions if ds.split_of[t.transaction_id] == "train"
    }
    covered = len(train_accounts & warmup_accounts) / len(train_accounts) if train_accounts else 0.0

    rep.measure("warmup_rows", len(warmup))
    rep.measure("warmup_span_days", span_days)
    rep.measure("warmup_days_configured", configured)
    rep.measure("warmup_account_coverage_of_train", round(covered, 4))

    if configured < LONGEST_FEATURE_WINDOW_DAYS:
        rep.error(
            "warmup_too_short",
            f"warm-up is {configured} days, shorter than the "
            f"{LONGEST_FEATURE_WINDOW_DAYS}-day feature window",
        )
    if span_days < LONGEST_FEATURE_WINDOW_DAYS:
        rep.error(
            "warmup_too_short",
            f"warm-up traffic spans only {span_days} days",
        )
    if covered < 0.5:
        rep.warn(
            "warmup_thin_coverage",
            f"only {covered:.1%} of TRAIN accounts have any warm-up history, so their "
            "rolling features start cold",
        )


# --------------------------------------------------------------------------
# §6 Leakage pre-checks
# --------------------------------------------------------------------------


def check_ground_truth_quarantine(rep: ValidationReport) -> None:
    """No feature, model, or scoring module may reach the registry (§6).

    An AST scan, not a text search: ``ml/feature_contract.py`` legitimately contains
    the string ``"is_fraud"`` as a dictionary key that classifies it as a label, and
    several docstrings discuss the field by name. What is forbidden is an *import* of
    the ground-truth module and an *attribute read* such as ``row.is_fraud``.
    """
    scanned = 0
    offences: List[str] = []
    for package in QUARANTINED_PACKAGES:
        root = REPO_ROOT / package
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(REPO_ROOT).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("simulator.ground_truth"):
                            offences.append(f"{relative}:{node.lineno} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").startswith("simulator.ground_truth"):
                        offences.append(f"{relative}:{node.lineno} imports from {node.module}")
                elif isinstance(node, ast.Attribute) and node.attr in LABEL_FIELDS:
                    offences.append(f"{relative}:{node.lineno} reads .{node.attr}")

    rep.measure("quarantine_modules_scanned", scanned)
    rep.measure("quarantine_packages", [p for p in QUARANTINED_PACKAGES if (REPO_ROOT / p).is_dir()])
    for offence in offences:
        rep.error("ground_truth_quarantine_breach", offence)


def _pit_signature(view, txn) -> Tuple:
    """The reductions a Checkpoint 2 feature would read, as one comparable tuple."""
    key, value = "account_id", txn.account_id
    previous = view.previous(key=key, value=value)
    return (
        view.count(key=key, value=value),
        view.count(timedelta(minutes=15), key=key, value=value),
        view.count(timedelta(days=7), key=key, value=value),
        round(view.amount_sum(timedelta(days=1), key=key, value=value), 6),
        round(view.deviation(txn.amount, key=key, value=value), 9),
        previous["transaction_id"] if previous else None,
        view.distance_from_km(txn.lat, txn.lon, key=key, value=value),
    )


def check_point_in_time_at_scale(
    ds: GeneratedDataset, rep: ValidationReport, *, verbose: bool = False
) -> None:
    """Recompute-after-truncation on real rows (``ML_READINESS`` §6, §7).

    The unit tests in ``tests/leakage/`` prove the mechanism on a dozen hand-built
    rows. This proves it on the real dataset, and it is the same experiment inverted:
    build the history from *all* transactions, build a second history from only the
    rows before a pivot, and recompute the features for sampled transactions that
    precede the pivot. If any feature differs, a later row reached backwards.

    Two histories rather than one per sampled row, because a full rebuild costs
    seconds and 4,000 of them would cost hours for no extra assurance.
    """
    txns = sorted(ds.transactions, key=lambda t: (t.timestamp, t.transaction_id))
    if len(txns) < 100:
        rep.error("dataset_too_small", "fewer than 100 transactions to verify")
        return

    pivot_index = int(len(txns) * 0.6)
    pivot = txns[pivot_index].timestamp
    earlier = [t for t in txns if t.timestamp < pivot]

    if verbose:
        print(f"  building history over {len(txns)} rows ...", flush=True)
    full = TransactionHistory.from_transactions(txns)
    truncated = TransactionHistory.from_transactions(earlier)

    rng = random.Random(PIT_SAMPLE_SEED)
    # Sampled from rows strictly before the pivot, excluding the very first instant:
    # those rows have no history in either store, so they could not reveal a
    # difference and would only dilute the sample.
    first_instant = earlier[0].timestamp
    candidates = [t for t in earlier if t.timestamp > first_instant]
    sample = (
        rng.sample(candidates, PIT_SAMPLE_SIZE)
        if len(candidates) > PIT_SAMPLE_SIZE
        else candidates
    )

    mismatches: List[str] = []
    boundary_violations = 0
    self_inclusions = 0
    non_finite = 0
    for txn in sample:
        view_full = full.view_for(txn)
        view_truncated = truncated.view_for(txn)

        if _pit_signature(view_full, txn) != _pit_signature(view_truncated, txn):
            mismatches.append(txn.transaction_id)

        rows = view_full.rows(key="account_id", value=txn.account_id)
        if any(r["timestamp"] >= txn.timestamp for r in rows):
            boundary_violations += 1
        if any(r["transaction_id"] == txn.transaction_id for r in rows):
            self_inclusions += 1
        deviation = view_full.deviation(txn.amount, key="account_id", value=txn.account_id)
        if deviation != deviation or deviation in (float("inf"), float("-inf")):
            non_finite += 1

    # Exact-timestamp collisions must actually occur, or the tie-exclusion rule is
    # untested by this dataset and the unit test is the only evidence for it.
    stamps = Counter(t.timestamp for t in txns)
    collisions = sum(c - 1 for c in stamps.values() if c > 1)

    rep.measure("pit_rows_in_history", len(txns))
    rep.measure("pit_pivot_timestamp", pivot.isoformat())
    rep.measure("pit_rows_sampled", len(sample))
    rep.measure("pit_feature_mismatches", len(mismatches))
    rep.measure("pit_boundary_violations", boundary_violations)
    rep.measure("pit_self_inclusions", self_inclusions)
    rep.measure("pit_non_finite_features", non_finite)
    rep.measure("pit_exact_timestamp_collisions", collisions)

    if mismatches:
        rep.error(
            "point_in_time_contamination",
            f"{len(mismatches)} of {len(sample)} sampled rows changed when later "
            f"transactions were added (e.g. {mismatches[:3]})",
        )
    if boundary_violations:
        rep.error(
            "point_in_time_boundary",
            f"{boundary_violations} sampled views contained a row at or after their as-of instant",
        )
    if self_inclusions:
        rep.error(
            "point_in_time_self_inclusion",
            f"{self_inclusions} sampled transactions appeared in their own history",
        )
    if non_finite:
        rep.error(
            "point_in_time_non_finite",
            f"{non_finite} sampled rows produced a non-finite deviation",
        )
    if collisions == 0:
        rep.warn(
            "no_timestamp_collisions",
            "no two transactions share a timestamp, so the tie-exclusion rule is "
            "exercised only by the unit tests",
        )


# --------------------------------------------------------------------------
# §8 Graph readiness
# --------------------------------------------------------------------------


class _UnionFind:
    """Union-find over string node IDs.

    Hand-rolled on purpose: NetworkX is a Checkpoint 3 dependency
    (``pyproject.toml`` ``[graph]``) and the readiness gate must run before it is
    installed. Path halving plus union by size, which is all connectivity needs.
    """

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node: str) -> str:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return groups


def _find_cycles(adjacency: Dict[str, set]) -> set:
    """Nodes lying on a directed cycle, by iterative DFS with three colours.

    Iterative because a 156k-row dataset can produce a chain deeper than the
    recursion limit, and a readiness audit that dies of a ``RecursionError`` is
    indistinguishable from a readiness audit that found nothing.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {n: WHITE for n in adjacency}
    on_cycle: set = set()

    for start in adjacency:
        if colour[start] != WHITE:
            continue
        stack: List[Tuple[str, Iterable[str]]] = [(start, iter(adjacency[start]))]
        path: List[str] = [start]
        position: Dict[str, int] = {start: 0}
        colour[start] = GREY
        while stack:
            node, neighbours = stack[-1]
            advanced = False
            for neighbour in neighbours:
                if neighbour not in colour:
                    continue
                if colour[neighbour] == GREY:
                    on_cycle.update(path[position[neighbour]:])
                elif colour[neighbour] == WHITE:
                    colour[neighbour] = GREY
                    position[neighbour] = len(path)
                    path.append(neighbour)
                    stack.append((neighbour, iter(adjacency[neighbour])))
                    advanced = True
                    break
            if not advanced:
                colour[node] = BLACK
                stack.pop()
                path.pop()
                position.pop(node, None)
    return on_cycle


def check_graph_readiness(ds: GeneratedDataset, rep: ValidationReport) -> None:
    """The graph must be worth analysing (``ML_READINESS`` §8).

    The graph is built from *observed transactions* plus beneficiary links, not from
    the generator's relationship pools — a detector at Checkpoint 3 sees the former
    and never the latter, so measuring the latter would flatter the dataset.
    """
    uf = _UnionFind()
    degree: Counter = Counter()
    edges: set = set()
    device_accounts: Dict[str, set] = defaultdict(set)
    ip_accounts: Dict[str, set] = defaultdict(set)
    account_neighbours: Dict[str, set] = defaultdict(set)
    entity_accounts: Dict[str, set] = defaultdict(set)

    def link(a: str, b: str) -> None:
        if (a, b) in edges:
            return
        edges.add((a, b))
        uf.union(a, b)
        degree[a] += 1
        degree[b] += 1

    for t in ds.transactions:
        account = f"acct:{t.account_id}"
        uf.add(account)
        for prefix, value in (
            ("dev", t.device_id),
            ("ip", t.ip_id),
            ("mer", t.merchant_id),
            ("ben", t.beneficiary_id),
        ):
            if value is None:
                continue
            node = f"{prefix}:{value}"
            link(account, node)
            account_neighbours[t.account_id].add(node)
            entity_accounts[node].add(t.account_id)
        if t.device_id:
            device_accounts[t.device_id].add(t.account_id)
        if t.ip_id:
            ip_accounts[t.ip_id].add(t.account_id)

    # The edge that closes a money-flow cycle.
    money_flow: Dict[str, set] = defaultdict(set)
    for beneficiary in ds.ecosystem.beneficiaries:
        if beneficiary.linked_account_id:
            link(f"ben:{beneficiary.beneficiary_id}", f"acct:{beneficiary.linked_account_id}")
    beneficiary_target = {
        b.beneficiary_id: b.linked_account_id
        for b in ds.ecosystem.beneficiaries
        if b.linked_account_id
    }
    for t in ds.transactions:
        target = beneficiary_target.get(t.beneficiary_id) if t.beneficiary_id else None
        if target and target != t.account_id:
            money_flow[t.account_id].add(target)

    components = uf.components()
    sizes = sorted((len(v) for v in components.values()), reverse=True)
    node_count = len(uf.parent)
    largest_share = sizes[0] / node_count if node_count else 0.0
    isolated = sum(1 for n in uf.parent if degree[n] == 0)

    rep.measure("graph_nodes", node_count)
    rep.measure("graph_edges", len(edges))
    rep.measure("graph_components", len(components))
    rep.measure("graph_largest_component_share", round(largest_share, 4))
    rep.measure("graph_isolated_nodes", isolated)

    if largest_share < 0.5:
        rep.error(
            "graph_fragmented",
            f"the largest connected component holds only {largest_share:.1%} of nodes",
        )
    if isolated > node_count * 0.05:
        rep.error(
            "graph_isolated_nodes",
            f"{isolated} of {node_count} nodes have no edges",
        )

    # Degree distribution: degenerate means all-equal or one hub and nothing else.
    account_degrees = sorted(degree[n] for n in uf.parent if n.startswith("acct:"))
    if account_degrees:
        median = account_degrees[len(account_degrees) // 2]
        rep.measure(
            "graph_account_degree",
            {
                "min": account_degrees[0],
                "median": median,
                "p90": account_degrees[int(len(account_degrees) * 0.9)],
                "max": account_degrees[-1],
                "distinct_values": len(set(account_degrees)),
            },
        )
        if len(set(account_degrees)) < 4:
            rep.error(
                "graph_degenerate_degrees",
                f"account degree takes only {len(set(account_degrees))} distinct values",
            )
        if account_degrees[-1] <= median:
            rep.error("graph_degenerate_degrees", "no account is better connected than the median")

    # Legitimate sharing baseline: benign overlap the rings must hide inside.
    fraud_accounts = {t.account_id for t in ds.transactions if t.is_fraud}
    benign_shared_devices = sum(
        1 for accs in device_accounts.values() if len(accs) > 1 and not (accs & fraud_accounts)
    )
    benign_shared_ips = sum(
        1 for accs in ip_accounts.values() if len(accs) > 1 and not (accs & fraud_accounts)
    )
    rep.measure("graph_benign_shared_devices", benign_shared_devices)
    rep.measure("graph_benign_shared_ips", benign_shared_ips)
    if benign_shared_devices == 0:
        rep.error(
            "no_benign_device_sharing",
            "every multi-account device involves a fraudulent account, so device "
            "sharing alone identifies fraud",
        )
    if benign_shared_ips == 0:
        rep.error(
            "no_benign_ip_sharing",
            "every multi-account IP involves a fraudulent account, so IP sharing "
            "alone identifies fraud",
        )

    # Rings must be embedded: a ring whose shared entities are touched only by ring
    # members is a separate component, and a connectivity filter would find it.
    coordinated = [r for r in ds.ground_truth.records if r.detection_class == "B"]
    isolated_rings: List[str] = []
    for record in coordinated:
        ring = set(record.affected_account_ids)
        if len(ring) < 2:
            continue
        neighbours: set = set()
        for account in ring:
            neighbours |= account_neighbours.get(account, set())
        outsiders = set()
        for node in neighbours:
            outsiders |= entity_accounts.get(node, set()) - ring
        if not outsiders:
            isolated_rings.append(record.attack_id)

    ring_total = sum(1 for r in coordinated if len(set(r.affected_account_ids)) >= 2)
    rep.measure("graph_coordinated_rings", ring_total)
    rep.measure("graph_isolated_rings", len(isolated_rings))
    if isolated_rings:
        rep.error(
            "rings_trivially_isolated",
            f"{len(isolated_rings)} of {ring_total} multi-account rings share no entity "
            f"with any account outside the ring (e.g. {isolated_rings[:3]})",
        )

    # Cycles: circular money flow must actually close.
    on_cycle = _find_cycles({k: set(v) for k, v in money_flow.items()})
    circular = [r for r in ds.ground_truth.records if r.attack_type == "circular_money_flow"]
    with_cycle = sum(1 for r in circular if set(r.affected_account_ids) & on_cycle)

    rep.measure("graph_money_flow_edges", sum(len(v) for v in money_flow.values()))
    rep.measure("graph_accounts_on_cycle", len(on_cycle))
    rep.measure("graph_circular_attacks", len(circular))
    rep.measure("graph_circular_attacks_with_cycle", with_cycle)

    if circular and with_cycle == 0:
        rep.error(
            "no_detectable_cycles",
            f"{len(circular)} circular_money_flow attacks exist but none produced a "
            "cycle in the account→beneficiary→account graph",
        )
    elif circular and with_cycle < len(circular) * 0.5:
        rep.warn(
            "few_detectable_cycles",
            f"only {with_cycle} of {len(circular)} circular_money_flow attacks close a cycle",
        )


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def run_audit(
    ds: GeneratedDataset, *, verbose: bool = False
) -> Tuple[ValidationReport, ValidationReport, Any]:
    """Run every section. Returns ``(gap_checks, checkpoint_1_validation, shortcuts)``."""
    if verbose:
        print("validating dataset (Checkpoint 1 checks) ...", flush=True)
    validation = validate_dataset(ds)

    if verbose:
        print("auditing shortcuts (Checkpoint 1 probes) ...", flush=True)
    shortcuts = audit_shortcuts(ds, verbose=verbose)

    rep = ValidationReport()
    for name, check in (
        ("duplicate rows", lambda: check_duplicate_rows(ds, rep)),
        ("class balance per split", lambda: check_both_classes_present(ds, rep)),
        ("fraud temporal spread", lambda: check_fraud_not_time_confined(ds, rep)),
        ("warm-up sufficiency", lambda: check_warmup_sufficiency(ds, rep)),
        ("ground-truth quarantine", lambda: check_ground_truth_quarantine(rep)),
        (
            "point-in-time safety at scale",
            lambda: check_point_in_time_at_scale(ds, rep, verbose=verbose),
        ),
        ("graph readiness", lambda: check_graph_readiness(ds, rep)),
    ):
        if verbose:
            print(f"checking {name} ...", flush=True)
        check()

    return rep, validation, shortcuts


#: ``(section, source)`` for the gate table, in ``ML_READINESS`` order.
SECTION_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("§2 dataset sanity", "checkpoint_1_validation + duplicate check"),
    ("§3 triviality", "shortcut audit"),
    ("§4 class balance", "checkpoint_1_validation + single-class check"),
    ("§5 temporal distribution", "checkpoint_1_validation + spread and warm-up checks"),
    ("§6 leakage pre-checks", "quarantine scan + point-in-time at scale"),
    ("§7 determinism", "manifest fingerprint (tests/simulator/test_reproducibility.py)"),
    ("§8 graph readiness", "graph check"),
)


def build_payload(
    ds: GeneratedDataset,
    gaps: ValidationReport,
    validation: ValidationReport,
    shortcuts: Any,
) -> Dict[str, Any]:
    """One JSON-serialisable record of the whole audit."""
    return {
        "seed": ds.config.seed,
        "prevalence_configured": ds.config.fraud_prevalence,
        "prevalence_label": "primary synthetic benchmark configuration"
        if ds.config.fraud_prevalence == 0.05
        else "synthetic benchmark configuration",
        "manifest": ds.manifest,
        "gate": {
            "overall_pass": gaps.ok and validation.ok and shortcuts.ok,
            "checkpoint_1_validation": validation.ok,
            "shortcut_audit": shortcuts.ok,
            "readiness_gap_checks": gaps.ok,
            "sections": [{"section": s, "source": src} for s, src in SECTION_SOURCES],
        },
        "checkpoint_1_validation": validation.to_dict(),
        "shortcut_audit": shortcuts.to_dict(),
        "readiness_checks": gaps.to_dict(),
        "notes": [
            "No model was trained. PR-AUC and ROC-AUC are Not yet measured.",
            "No threshold was selected. The held-out split was not opened.",
            "Cost figures elsewhere are synthetic benchmark assumptions.",
        ],
    }


def _prevalence(raw: str) -> float:
    value = float(raw)
    if value not in SUPPORTED_PREVALENCES:
        raise argparse.ArgumentTypeError(
            f"prevalence must be one of {SUPPORTED_PREVALENCES}, got {value}"
        )
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Checkpoint 1.5 ML-readiness gate on a freshly generated dataset.",
        epilog=(
            "One prevalence per invocation, because each needs its own dataset. "
            "Trains nothing and never opens the held-out split."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="master seed (default: 42)")
    parser.add_argument(
        "--prevalence",
        type=_prevalence,
        default=0.05,
        help=f"fraud prevalence, one of {SUPPORTED_PREVALENCES} (default: 0.05)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    config = default_config(seed=args.seed, prevalence=args.prevalence)
    if verbose:
        print(
            f"generating dataset (seed={config.seed}, "
            f"prevalence={config.fraud_prevalence}) ...",
            flush=True,
        )
    ds = generate_dataset(config)

    gaps, validation, shortcuts = run_audit(ds, verbose=verbose)
    payload = build_payload(ds, gaps, validation, shortcuts)

    print()
    print("=" * 72)
    print(f"ML-READINESS AUDIT — seed {config.seed}, prevalence {config.fraud_prevalence}")
    print("=" * 72)
    print(f"transactions:       {len(ds.transactions)}")
    print(f"fraud:              {sum(1 for t in ds.transactions if t.is_fraud)}")
    print(f"attacks:            {len(ds.ground_truth)}")
    print()
    for key in sorted(gaps.measurements):
        print(f"  {key}: {gaps.measurements[key]}")
    print()
    print(f"checkpoint 1 validation: {validation.summary()}")
    print(f"shortcut audit:          {shortcuts.summary()}")
    print(f"readiness gap checks:    {gaps.summary()}")

    for label, report in (("validation", validation), ("readiness", gaps)):
        for issue in report.warnings:
            print(f"  warning ({label}): {issue.code}: {issue.message}")
        for issue in report.errors:
            print(f"  ERROR   ({label}): {issue.code}: {issue.message}")
    for result in shortcuts.oracles:
        print(f"  ORACLE: {result.feature} -> {result.target} (auc={result.auc})")

    passed = payload["gate"]["overall_pass"]
    print()
    print(f"GATE: {'PASS' if passed else 'FAIL'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        print(f"wrote: {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
