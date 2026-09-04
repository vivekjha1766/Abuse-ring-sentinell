"""Ground-truth registry — the authoritative record of what actually happened.

**This module is evaluation-only.** It knows exactly which transactions and
entities are fraudulent, which makes it the single most dangerous import in the
codebase. No feature, model, scorer, reputation calculation, or graph-scoring
module may import it, directly or transitively
(``docs/FEATURE_LEAKAGE_POLICY.md`` §4). That rule is enforced by an
import-level test rather than by convention, and this module is deliberately
kept free of any dependency on feature or model code so the quarantine boundary
stays easy to check.

The registry is written by the simulator and read only by the evaluation engine.
Its field set is designed to support all four evaluation levels in
``EVALUATION.md``:

* **transaction level** — ``affected_transaction_ids``
* **entity level** — the six ``affected_*_ids`` collections
* **ring level** — those collections treated as a group
* **temporal level** — ``start_time`` versus the first attributable alert
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ATTACK_FAMILIES",
    "AttackRecord",
    "GroundTruthRegistry",
    "Severity",
]

#: The seven attack families (DATA_DICTIONARY.md §5, PRD FR-2).
ATTACK_FAMILIES: Tuple[str, ...] = (
    "fake_merchant_ring",
    "shared_device_ring",
    "transaction_spike",
    "circular_money_flow",
    "synthetic_identity_cluster",
    "microtransaction_attack",
    "account_takeover",
)

#: Class A = individual/transaction anomalies, Class B = coordinated abuse.
#: Results are reported separately per class (``EVALUATION.md`` §6); this map is
#: the single definition of which family belongs to which class.
DETECTION_CLASS: Mapping[str, str] = {
    "fake_merchant_ring": "B",
    "shared_device_ring": "B",
    "transaction_spike": "A",
    "circular_money_flow": "B",
    "synthetic_identity_cluster": "B",
    "microtransaction_attack": "B",
    "account_takeover": "A",
}

Severity = str
SEVERITIES: Tuple[Severity, ...] = ("low", "medium", "high", "critical")


@dataclass(frozen=True)
class AttackRecord:
    """One attack instance and everything it touched (DATA_DICTIONARY.md §4)."""

    attack_id: str
    attack_type: str
    start_time: datetime
    end_time: datetime
    severity: Severity
    generation_parameters: Dict[str, Any]
    split: str
    """Which temporal split the attack was injected into."""
    affected_transaction_ids: List[str] = field(default_factory=list)
    affected_customer_ids: List[str] = field(default_factory=list)
    affected_merchant_ids: List[str] = field(default_factory=list)
    affected_account_ids: List[str] = field(default_factory=list)
    affected_device_ids: List[str] = field(default_factory=list)
    affected_ip_ids: List[str] = field(default_factory=list)
    affected_beneficiary_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.attack_type not in ATTACK_FAMILIES:
            raise ValueError(f"unknown attack_type: {self.attack_type!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity!r}")
        if self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")

    @property
    def detection_class(self) -> str:
        """``"A"`` (individual anomaly) or ``"B"`` (coordinated abuse)."""
        return DETECTION_CLASS[self.attack_type]

    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 3600.0

    def entity_ids(self) -> Dict[str, List[str]]:
        """All affected entity collections, keyed by entity type."""
        return {
            "customer": list(self.affected_customer_ids),
            "merchant": list(self.affected_merchant_ids),
            "account": list(self.affected_account_ids),
            "device": list(self.affected_device_ids),
            "ip": list(self.affected_ip_ids),
            "beneficiary": list(self.affected_beneficiary_ids),
        }

    def ring_entity_set(self) -> set:
        """Flat set of every entity in the attack — the ring-level truth set.

        Used for the Jaccard overlap computation in ``EVALUATION.md`` §4.
        """
        out: set = set()
        for ids in self.entity_ids().values():
            out.update(ids)
        return out

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        data["detection_class"] = self.detection_class
        return data


class GroundTruthRegistry:
    """Collection of :class:`AttackRecord` with evaluation-oriented lookups."""

    def __init__(self, records: Optional[Iterable[AttackRecord]] = None) -> None:
        self._records: List[AttackRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> List[AttackRecord]:
        return list(self._records)

    def add(self, record: AttackRecord) -> None:
        if any(r.attack_id == record.attack_id for r in self._records):
            raise ValueError(f"duplicate attack_id: {record.attack_id}")
        self._records.append(record)

    def extend(self, records: Iterable[AttackRecord]) -> None:
        for record in records:
            self.add(record)

    # -- lookups ----------------------------------------------------------
    def by_id(self, attack_id: str) -> AttackRecord:
        for record in self._records:
            if record.attack_id == attack_id:
                return record
        raise KeyError(f"no such attack_id: {attack_id}")

    def by_family(self, family: str) -> List[AttackRecord]:
        return [r for r in self._records if r.attack_type == family]

    def by_split(self, split: str) -> List[AttackRecord]:
        return [r for r in self._records if r.split == split]

    def by_class(self, detection_class: str) -> List[AttackRecord]:
        return [r for r in self._records if r.detection_class == detection_class]

    def fraudulent_transaction_ids(self) -> set:
        out: set = set()
        for record in self._records:
            out.update(record.affected_transaction_ids)
        return out

    def transaction_to_attack(self) -> Dict[str, str]:
        """Map each fraudulent transaction to the attack that produced it."""
        mapping: Dict[str, str] = {}
        for record in self._records:
            for txn_id in record.affected_transaction_ids:
                mapping[txn_id] = record.attack_id
        return mapping

    def family_counts(self) -> Dict[str, int]:
        return {f: len(self.by_family(f)) for f in ATTACK_FAMILIES}

    def transaction_counts_by_family(self) -> Dict[str, int]:
        return {
            f: sum(len(r.affected_transaction_ids) for r in self.by_family(f))
            for f in ATTACK_FAMILIES
        }

    def coverage_summary(self) -> Dict[str, Any]:
        """Per-split, per-family instance counts — recorded in the manifest.

        Used by the Checkpoint 1.5 audit to confirm every family appears in
        every split (``docs/ML_READINESS.md`` §5).
        """
        summary: Dict[str, Any] = {}
        for split in sorted({r.split for r in self._records}):
            summary[split] = {
                "attacks": len(self.by_split(split)),
                "fraud_transactions": sum(
                    len(r.affected_transaction_ids) for r in self.by_split(split)
                ),
                "families": {
                    f: len([r for r in self.by_split(split) if r.attack_type == f])
                    for f in ATTACK_FAMILIES
                },
            }
        return summary

    # -- persistence ------------------------------------------------------
    def to_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def write_json(self, path: str) -> None:
        """Write the registry with stable key ordering (byte-reproducible)."""
        payload = {"attacks": self.to_records()}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def read_json(cls, path: str) -> "GroundTruthRegistry":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records: List[AttackRecord] = []
        for raw in payload["attacks"]:
            raw = dict(raw)
            raw.pop("detection_class", None)
            raw["start_time"] = datetime.fromisoformat(raw["start_time"])
            raw["end_time"] = datetime.fromisoformat(raw["end_time"])
            records.append(AttackRecord(**raw))
        return cls(records)


def validate_record_completeness(record: AttackRecord) -> Sequence[str]:
    """Return a list of completeness problems, empty if the record is sound.

    FR-3 requires every record to support all four evaluation levels, which
    means at minimum: it names transactions, it names at least one entity, and
    it captures the parameters that generated it.
    """
    problems: List[str] = []
    if not record.affected_transaction_ids:
        problems.append(f"{record.attack_id}: no affected transactions")
    if not record.ring_entity_set():
        problems.append(f"{record.attack_id}: no affected entities")
    if not record.generation_parameters:
        problems.append(f"{record.attack_id}: empty generation_parameters")
    if record.end_time < record.start_time:
        problems.append(f"{record.attack_id}: end_time precedes start_time")
    if len(set(record.affected_transaction_ids)) != len(record.affected_transaction_ids):
        problems.append(f"{record.attack_id}: duplicate transaction ids")
    return problems
