"""Transaction spike (Class A — individual anomaly).

A sudden burst of activity on a single account: many more transactions in a
short window than that account normally makes, at somewhat larger amounts.

The defining design decision here is what this attack *does not* do. It uses the
account's **own** devices, **own** IPs, and its normal home location, and it buys
from ordinary merchants sampled with normal popularity weighting. Nothing about
the infrastructure is unusual. That is deliberate: it forces the detector to work
from velocity and amount deviation relative to the account's own baseline, which
is the genuine Class A signal. If a spike also came with a fresh device, it would
be indistinguishable from an account takeover and the two families would collapse
into one detection problem.

Because the burst is measured *relative to the account's own baseline*
(``multiplier`` × baseline frequency), a spike on a high-activity account is a
much larger absolute count than one on a quiet account. A global "count > N"
threshold therefore cannot separate spikes from ordinary busy accounts — which is
exactly the property ``docs/ML_READINESS.md`` §3 requires.
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
    jittered_amount,
    merchant_sampler,
    pick_accounts,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.config import TransactionSpikeParams

FAMILY = "transaction_spike"


def generate(ctx: AttackContext, params: TransactionSpikeParams) -> AttackResult:
    window = attack_window(ctx, params.duration_hours)
    collector = EntityCollector()
    transactions: List = []

    # A spike is only meaningful on an account with an established baseline to
    # deviate from, so require some prior history.
    candidates = pick_accounts(ctx, 1, min_activity=8)
    if not candidates:  # pragma: no cover - defensive
        record = build_record(
            ctx, family=FAMILY, window=window, severity="low",
            params=asdict(params), transactions=[], collector=collector,
        )
        return AttackResult([], record)
    account = candidates[0]

    # Volume relative to this account's own normal rate. The burst compresses
    # ``multiplier`` days' worth of that account's usual volume into
    # ``duration_hours``, so the *rate* amplification is
    # ``multiplier * 24 / duration_hours``. Expressing it against the account's
    # own daily baseline is what keeps a global "count > N" rule useless: a spike
    # on a quiet account is a smaller absolute count than an ordinary day for a
    # busy one.
    # ``min_transactions`` is a floor, and for a quiet victim it binds — which
    # would otherwise make every such spike exactly ``min_transactions`` rows
    # long. Jittering keeps the floor meaningful without turning it into a
    # constant signature.
    baseline_daily = max(0.0, account.baseline_tx_frequency)
    n_tx = vary_count(
        ctx,
        max(params.min_transactions, int(round(baseline_daily * params.multiplier))),
        spread=0.2,
    )

    pick_merchant = merchant_sampler(ctx)
    times = spread_timestamps(ctx, window, n_tx)
    mean = account.baseline_amount_mean * params.amount_scale
    std = account.baseline_amount_std * params.amount_scale

    for ts in times:
        device_id, ip_id = account_devices_ips(ctx, account, ts)
        lat, lon = account_home(ctx, account)
        txn = emit(
            ctx, timestamp=ts, account=account, amount=jittered_amount(ctx, mean, std),
            device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
            transaction_type="purchase", merchant_id=pick_merchant(ts),
        )
        transactions.append(txn)
        collector.add_transaction_entities(txn)

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    severity = severity_for(min(1.0, params.multiplier / 12.0 + params.amount_scale / 8.0))
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record)
