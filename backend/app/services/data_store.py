"""Loads non-heldout transaction/account data for the demo API.

Reads only data/splits/{warmup,train,validation} -- heldout is never named
or read anywhere in this module. If the generated dataset is not present
locally (`make data` has not been run in this environment), falls back to
a small, clearly-labelled synthetic fixture so the API still starts and
responds for a demo.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from backend.app.config import DATA_ROOT, PERMITTED_DATA_SPLITS
from simulator.transactions import Transaction


def _transaction(row: dict[str, str]) -> Transaction:
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
        is_fraud=row["is_fraud"].strip().lower() in {"true", "1"},
        attack_id=row["attack_id"] or None,
    )


def _read_split(split: str) -> list[Transaction]:
    if split not in PERMITTED_DATA_SPLITS:
        raise ValueError(f"unpermitted split for the demo API: {split}")
    path = DATA_ROOT / "splits" / split / "transactions.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [_transaction(row) for row in csv.DictReader(handle)]


def _read_accounts() -> dict[str, dict]:
    path = DATA_ROOT / "raw" / "accounts.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["account_id"]: {"created_at": datetime.fromisoformat(row["created_at"])}
            for row in csv.DictReader(handle)
        }


def _synthetic_fallback() -> tuple[list[Transaction], dict[str, dict]]:
    """A handful of hand-built, explicitly-illustrative demo rows.

    Used only when the real generated dataset is not present locally --
    the dataset is git-ignored and produced by `make data`, not shipped in
    this repo. Not derived from any real split; exists solely so the API
    and demo UI have something to show without requiring a local dataset
    to be generated first.
    """
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def txn(i: int, minutes: float, amount: float, account: str, device: str, ip: str) -> Transaction:
        return Transaction(
            transaction_id=f"demo_txn_{i:03d}",
            timestamp=origin + timedelta(minutes=minutes),
            account_id=account,
            customer_id=f"cus_{account}",
            merchant_id="mer_demo",
            beneficiary_id=None,
            device_id=device,
            ip_id=ip,
            amount=amount,
            currency="USD",
            channel="web",
            transaction_type="purchase",
            lat=10.0 + i * 0.01,
            lon=20.0 + i * 0.01,
            status="completed",
            is_fraud=False,
            attack_id=None,
        )

    transactions = [
        txn(0, 0, 42.50, "acc_demo_1", "dev_demo_1", "ip_demo_1"),
        txn(1, 15, 38.00, "acc_demo_1", "dev_demo_1", "ip_demo_1"),
        txn(2, 30, 1250.00, "acc_demo_1", "dev_demo_1", "ip_demo_1"),
        txn(3, 5, 45.00, "acc_demo_2", "dev_demo_1", "ip_demo_2"),
        txn(4, 20, 51.00, "acc_demo_3", "dev_demo_1", "ip_demo_2"),
    ]
    accounts = {
        "acc_demo_1": {"created_at": origin - timedelta(days=200)},
        "acc_demo_2": {"created_at": origin - timedelta(days=5)},
        "acc_demo_3": {"created_at": origin - timedelta(days=3)},
    }
    return transactions, accounts


class DemoData:
    def __init__(
        self,
        *,
        history: list[Transaction],
        accounts: dict[str, dict],
        by_id: dict[str, Transaction],
        is_synthetic_fallback: bool,
    ) -> None:
        self.history = history
        self.accounts = accounts
        self.by_id = by_id
        self.is_synthetic_fallback = is_synthetic_fallback

    def get(self, transaction_id: str) -> Transaction | None:
        return self.by_id.get(transaction_id)

    def sample_ids(self, n: int = 10) -> list[str]:
        return [t.transaction_id for t in self.history[:n]]


@lru_cache(maxsize=1)
def load_demo_data() -> DemoData:
    warmup = _read_split("warmup")
    train = _read_split("train")
    validation = _read_split("validation")
    accounts = _read_accounts()

    history = [*warmup, *train, *validation]
    is_synthetic_fallback = False
    if not history:
        history, accounts = _synthetic_fallback()
        is_synthetic_fallback = True

    by_id = {t.transaction_id: t for t in history}
    return DemoData(
        history=history, accounts=accounts, by_id=by_id, is_synthetic_fallback=is_synthetic_fallback
    )
