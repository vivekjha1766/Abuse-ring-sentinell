"""Benchmark the Checkpoint 2C feature path without opening heldout data."""

from __future__ import annotations

import csv
import json
import argparse
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.feature_registry import FEATURE_COLUMNS
from ml.features import build_feature_matrix
from simulator.transactions import Transaction


DATA_ROOT = REPO_ROOT / "data"
PERMITTED_SPLITS = ("warmup", "train", "validation")


def _transaction(row: dict[str, str]) -> Transaction:
    """Read only operational transaction columns; labels are never loaded."""
    return Transaction(
        transaction_id=row["transaction_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        account_id=row["account_id"],
        customer_id=row["customer_id"],
        merchant_id=row["merchant_id"] or None,
        beneficiary_id=row["beneficiary_id"] or None,
        device_id=row["device_id"],
        ip_id=row["ip_id"],
        amount=float(row["amount"]),
        currency=row["currency"],
        channel=row["channel"],
        transaction_type=row["transaction_type"],
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        status=row["status"],
        is_fraud=False,
        attack_id=None,
    )


def _read_transactions(split: str) -> list[Transaction]:
    if split not in PERMITTED_SPLITS:
        raise ValueError(f"unpermitted split: {split}")
    path = DATA_ROOT / "splits" / split / "transactions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [_transaction(row) for row in csv.DictReader(handle)]


def _read_accounts() -> dict[str, dict[str, datetime]]:
    path = DATA_ROOT / "raw" / "accounts.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["account_id"]: {"created_at": datetime.fromisoformat(row["created_at"])}
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    warmup = _read_transactions("warmup")
    train = _read_transactions("train")
    validation = _read_transactions("validation")
    accounts = _read_accounts()

    print("benchmarking train features", flush=True)
    tracemalloc.start()
    started = time.perf_counter()
    train_matrix = build_feature_matrix(
        train, history_transactions=[*warmup, *train], accounts=accounts
    )
    print("benchmarking validation features", flush=True)
    validation_matrix = build_feature_matrix(
        validation,
        history_transactions=[*warmup, *train, *validation],
        accounts=accounts,
    )
    elapsed_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "splits": {"train": len(train_matrix), "validation": len(validation_matrix)},
        "rows_processed": len(train_matrix) + len(validation_matrix),
        "feature_count": len(FEATURE_COLUMNS),
        "wall_clock_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_bytes,
        "heldout_accessed": False,
    }
    encoded = json.dumps(result, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
