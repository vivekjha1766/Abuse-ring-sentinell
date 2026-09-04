"""Microtransaction attack (Class B — coordinated abuse).

Very many very small payments — card testing, or value extracted in slices small
enough to fall under every review threshold. Individually these are the *least*
suspicious transactions in the dataset: a sub-currency-unit payment at a real
merchant is entirely unremarkable, and any amount-based rule will rank them at the
very bottom of its risk ordering.

That is precisely the point, and it is the reason this family sits in Class B. The
signal exists only in aggregate: an unusual concentration of tiny amounts, across
several accounts, converging on a small set of merchants in a short window. It is
also the family most likely to defeat a cost-sensitive threshold tuned on
transaction value, since each row's individual loss is negligible while the
aggregate is not — a tension examined in ``docs/COST_MODEL.md``.

Amounts are drawn uniformly from ``[amount_min, amount_max]`` and then rounded to
two decimals like every other amount in the dataset, so they are small but not
suspiciously *round*.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

from simulator.attacks.base import (
    AttackContext,
    AttackResult,
    EntityCollector,
    account_devices_ips,
    account_home,
    attack_window,
    build_record,
    emit,
    merchant_sampler,
    pick_accounts,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.base_generator import round_amount
from simulator.config import MicrotransactionParams

FAMILY = "microtransaction_attack"


def generate(ctx: AttackContext, params: MicrotransactionParams) -> AttackResult:
    rng = ctx.rng
    window = attack_window(ctx, params.duration_hours)
    collector = EntityCollector()
    transactions: List = []

    accounts = pick_accounts(ctx, vary_count(ctx, params.n_accounts, spread=0.2))
    if not accounts:  # pragma: no cover - defensive
        record = build_record(
            ctx, family=FAMILY, window=window, severity="low",
            params=asdict(params), transactions=[], collector=collector,
        )
        return AttackResult([], record)

    # The probing converges on a small merchant set — the coordination signal.
    # Its size scales with the ring so a larger ring spreads a little wider,
    # rather than being fixed at some arbitrary count. The set is fixed for the
    # whole attack, so every target must already exist when the window opens.
    pick_merchant = merchant_sampler(ctx)
    n_targets = max(1, len(accounts) // 4)
    targets = [pick_merchant(window.start) for _ in range(n_targets)]

    lo = min(params.amount_min, params.amount_max)
    hi = max(params.amount_min, params.amount_max)

    for account in accounts:
        # Per-account, so probing volume is uneven across the ring.
        n_tx = vary_count(ctx, params.tx_per_account, spread=0.3)
        times = spread_timestamps(ctx, window, n_tx)
        device_id, ip_id = account_devices_ips(ctx, account, window.start)
        for ts in times:
            amount = round_amount(float(rng.uniform(lo, hi)))
            lat, lon = account_home(ctx, account)
            txn = emit(
                ctx, timestamp=ts, account=account, amount=amount,
                device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
                transaction_type="purchase",
                merchant_id=targets[int(rng.integers(0, len(targets)))],
            )
            transactions.append(txn)
            collector.add_transaction_entities(txn)

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    # Severity reflects aggregate volume, not per-transaction value.
    severity = severity_for(min(1.0, 0.1 + len(transactions) / 800.0))
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record)
