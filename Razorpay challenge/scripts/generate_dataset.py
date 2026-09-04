#!/usr/bin/env python
"""Generate one synthetic dataset and write it to disk.

The single dataset entry point named by the specification (``ROADMAP.md``
Checkpoint 1 artifacts, ``DATA_DICTIONARY.md`` §7, ``README.md`` repository
layout). ``make data`` invokes it with no arguments.

**What it writes** — the lifecycle in ``DATA_DICTIONARY.md`` §7, nothing more::

    data/raw/        the whole dataset: transactions, six entity tables, the
                     link tables, the ground-truth registry, the manifest and
                     the validation report
    data/splits/     the same transactions partitioned by the temporal split,
                     one directory per split

**Determinism.** Seed and prevalence are the only inputs, both defaulted to the
primary synthetic benchmark configuration (seed 42, prevalence 0.05 — the values
recorded in ``.env.example``). They are read from the command line only, never
from the environment, so that an exported variable cannot silently change what
``make data`` produces. Cells are serialised with the same rules the manifest's
``content_hash`` uses, so two runs of the same seed and config produce
byte-identical files (``ROADMAP.md`` Checkpoint 1 exit criteria).

**The held-out seal.** ``docs/DATA_SPLIT_PROTOCOL.md`` §4 requires the held-out
split to live at its own path and requires this script never to import the
held-out loader. It does not: it writes each split from the in-memory partition
the pipeline already computed and reads nothing back.

Warm-up traffic gets its own directory alongside the three evaluation splits.
It is not a fourth split — it is never scored — but ``docs/DATA_SPLIT_PROTOCOL.md``
§5 exists so that 7-day rolling features have real history on the first TRAIN
day, and a feature builder can only get that history without touching the
held-out rows if warm-up is reachable separately from ``data/raw/``.

This script generates and persists. It does not audit: the shortcut audit and
the ML-readiness report belong to Checkpoint 1.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Type

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/generate_dataset.py`
    sys.path.insert(0, str(REPO_ROOT))

from simulator.config import SPLIT_NAMES, SUPPORTED_PREVALENCES, default_config
from simulator.entities import (
    Account,
    Beneficiary,
    Customer,
    Device,
    IPAddress,
    Merchant,
)
from simulator.pipeline import (
    WARMUP_SPLIT,
    GeneratedDataset,
    _field_repr,
    generate_dataset,
)
from simulator.transactions import TRANSACTION_FIELDS
from simulator.validation import validate_dataset

#: Every split that receives a directory: the three evaluation splits plus the
#: unscored warm-up history.
WRITTEN_SPLITS: Tuple[str, ...] = (WARMUP_SPLIT,) + tuple(SPLIT_NAMES)

#: ``(filename, Ecosystem attribute, dataclass)``. The dataclass supplies the
#: column order, so a new entity field appears in the CSV automatically instead
#: of being silently dropped by a hand-maintained header list.
ENTITY_TABLES: Tuple[Tuple[str, str, Type[Any]], ...] = (
    ("customers.csv", "customers", Customer),
    ("merchants.csv", "merchants", Merchant),
    ("accounts.csv", "accounts", Account),
    ("devices.csv", "devices", Device),
    ("ip_addresses.csv", "ips", IPAddress),
    ("beneficiaries.csv", "beneficiaries", Beneficiary),
)

#: The many-to-many relationships, written long-form so they load without
#: parsing embedded lists.
LINK_TABLES: Tuple[Tuple[str, str, Tuple[str, str]], ...] = (
    ("account_devices.csv", "account_devices", ("account_id", "device_id")),
    ("account_ips.csv", "account_ips", ("account_id", "ip_id")),
    (
        "account_beneficiaries.csv",
        "account_beneficiaries",
        ("account_id", "beneficiary_id"),
    ),
    ("customer_accounts.csv", "customer_accounts", ("customer_id", "account_id")),
)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def _cell(value: Any) -> str:
    """One CSV cell, canonicalised exactly as the manifest's hash canonicalises.

    :func:`simulator.pipeline._field_repr` is the private canonicaliser behind
    ``content_hash``: ``None`` becomes empty, datetimes become ISO-8601, booleans
    become ``1``/``0``, floats are fixed at six decimals. It is imported rather
    than restated on purpose. Restating the rules would let the stored file and
    the reproducibility metadata drift apart silently; delegating means
    ``data/raw/transactions.csv`` can be verified against
    ``manifest.json``'s ``content_hash`` by re-joining its cells with ``|``.
    """
    return _field_repr(value)


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    """Write a CSV with fixed newlines, returning the row count."""
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([_cell(v) for v in row])
            count += 1
    return count


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable key ordering, matching the registry's own writer."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar from a validation measurement
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)  # pragma: no cover - defensive


def _write_transactions(path: Path, transactions: Sequence[Any]) -> int:
    """Transactions in the canonical :data:`TRANSACTION_FIELDS` order.

    Label columns are included: ``DATA_DICTIONARY.md`` §2 defines them as part of
    the row, and the one-way barrier is enforced where features are built
    (:mod:`simulator.leakage`), not by mutilating the stored dataset.
    """
    return _write_csv(
        path,
        TRANSACTION_FIELDS,
        ([getattr(t, name) for name in TRANSACTION_FIELDS] for t in transactions),
    )


# --------------------------------------------------------------------------
# Writing one dataset
# --------------------------------------------------------------------------


def write_dataset(dataset: GeneratedDataset, out_dir: Path) -> Dict[str, Any]:
    """Persist ``dataset`` under ``out_dir``; return what was written."""
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    written: Dict[str, int] = {}
    written["transactions.csv"] = _write_transactions(
        raw / "transactions.csv", dataset.transactions
    )

    eco = dataset.ecosystem
    for filename, attribute, model in ENTITY_TABLES:
        columns = [f.name for f in dataclass_fields(model)]
        records = getattr(eco, attribute)
        written[filename] = _write_csv(
            raw / filename,
            columns,
            ([getattr(r, c) for c in columns] for r in records),
        )

    for filename, attribute, (left, right) in LINK_TABLES:
        mapping: Dict[str, List[str]] = getattr(eco, attribute)
        written[filename] = _write_csv(
            raw / filename,
            (left, right),
            (
                (key, value)
                for key in sorted(mapping)
                for value in mapping[key]
            ),
        )

    dataset.ground_truth.write_json(str(raw / "ground_truth.json"))
    _write_json(raw / "manifest.json", dataset.manifest)

    # Split partition. Written from the in-memory assignment the pipeline
    # computed — no loader is imported, and the held-out rows land only in
    # their own directory (docs/DATA_SPLIT_PROTOCOL.md §4).
    by_split: Dict[str, List[Any]] = {name: [] for name in WRITTEN_SPLITS}
    for txn in dataset.transactions:
        by_split[dataset.split_of[txn.transaction_id]].append(txn)

    split_rows: Dict[str, int] = {}
    for name in WRITTEN_SPLITS:
        split_dir = out_dir / "splits" / name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows[name] = _write_transactions(
            split_dir / "transactions.csv", by_split[name]
        )

    return {"raw": written, "splits": split_rows}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _prevalence(raw: str) -> float:
    value = float(raw)
    if value not in SUPPORTED_PREVALENCES:
        raise argparse.ArgumentTypeError(
            f"prevalence must be one of {SUPPORTED_PREVALENCES}, got {value}"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic Abuse-Ring Sentinel dataset.",
        epilog=(
            "Defaults are the primary synthetic benchmark configuration "
            "(a benchmark choice, not a real-world fraud rate)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="master seed (default: 42)"
    )
    parser.add_argument(
        "--prevalence",
        type=_prevalence,
        default=0.05,
        help=f"fraud prevalence, one of {SUPPORTED_PREVALENCES} (default: 0.05)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data",
        help="output directory (default: ./data)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    config = default_config(seed=args.seed, prevalence=args.prevalence)
    print(
        f"generating: seed={config.seed} prevalence={config.fraud_prevalence:.0%} "
        f"days={config.days} warmup_days={config.warmup_days}"
    )
    dataset = generate_dataset(config)

    report = validate_dataset(dataset)
    written = write_dataset(dataset, args.out)
    _write_json(args.out / "raw" / "validation.json", report.to_dict())

    manifest = dataset.manifest
    print(f"config_fingerprint: {manifest['config_fingerprint']}")
    print(f"content_hash:       {manifest['content_hash']}")
    print(f"rng_streams:        {manifest['rng_streams']}")
    print(f"entities:           {manifest['entity_counts']}")
    print(f"boundaries:         {manifest['boundaries']}")
    print(f"transactions:       {manifest['transactions']}")
    print(f"prevalence:         {manifest['prevalence']}")
    print(f"attacks:            {manifest['attacks']['total']} "
          f"{manifest['attacks']['by_family']}")
    print(f"rows written:       raw={written['raw']['transactions.csv']} "
          f"splits={written['splits']}")
    print(f"wrote:              {args.out}")

    for issue in report.warnings:
        print(f"  warning: {issue.code}: {issue.message}")
    for issue in report.errors:
        print(f"  ERROR:   {issue.code}: {issue.message}")
    print(f"validation: {report.summary()}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
