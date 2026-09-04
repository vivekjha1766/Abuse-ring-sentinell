"""Circular money flow (Class B — coordinated abuse).

Funds move A → B → C → … → A, returning close to their origin, with a little value
skimmed at each hop. No single transfer looks wrong: each is one account paying a
beneficiary, an entirely ordinary act. The abuse is only visible as a *cycle* in
the transfer graph, which is why this family is the clearest justification for the
graph layer in Checkpoint 3 — a purely row-wise model has essentially nothing to
work with here.

Closing the cycle requires beneficiaries whose ``linked_account_id`` points at the
next account in the ring, so money paid to a beneficiary lands back in an account
that can pay onward. Such links also occur legitimately (paying your own savings
account), so the link itself is not suspicious
(``simulator/entities.py::_generate_beneficiaries``). Existing linked
beneficiaries are reused where one already points at the needed account; only the
missing links are created.

Hops are strictly ordered in time. A cycle whose transfers were shuffled would be
detectable as a cycle but would not represent money actually flowing, and the
temporal-ordering constraint is what makes cycle detection non-trivial.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List

from simulator.attacks.base import (
    AttackContext,
    AttackResult,
    EntityCollector,
    account_devices_ips,
    account_home,
    attack_window,
    build_record,
    emit,
    entity_created_at,
    pick_accounts,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.base_generator import round_amount
from simulator.config import CircularMoneyFlowParams
from simulator.entities import BENEFICIARY_TYPES

FAMILY = "circular_money_flow"


def _beneficiary_for(ctx: AttackContext, account_id: str, not_after: datetime) -> str:
    """A beneficiary that pays into ``account_id``, reusing one if it exists.

    Reuse is restricted to beneficiaries that already existed at ``not_after``.
    Without that guard the ring could route through a payee another attack
    registered later in the same split, producing transactions that reference a
    beneficiary before it was created (measured: 4 rows at -53.8 days).
    """
    for beneficiary in ctx.eco.beneficiaries:
        if (
            beneficiary.linked_account_id == account_id
            and beneficiary.created_at <= not_after
        ):
            return beneficiary.beneficiary_id
    rng = ctx.rng
    customer = ctx.eco.customer_by_id[ctx.eco.account_by_id[account_id].customer_id]
    created = entity_created_at(ctx, not_after=not_after, history_days=500.0)
    beneficiary = ctx.eco.add_beneficiary(
        created_at=created,
        beneficiary_type=BENEFICIARY_TYPES[int(rng.integers(0, len(BENEFICIARY_TYPES)))],
        country=customer.country,
        linked_account_id=account_id,
    )
    return beneficiary.beneficiary_id


def generate(ctx: AttackContext, params: CircularMoneyFlowParams) -> AttackResult:
    window = attack_window(ctx, params.duration_hours)
    collector = EntityCollector()
    transactions: List = []

    cycle_length = max(2, params.cycle_length)
    accounts = pick_accounts(ctx, cycle_length)
    if len(accounts) < 2:  # pragma: no cover - defensive
        record = build_record(
            ctx, family=FAMILY, window=window, severity="low",
            params=asdict(params), transactions=[], collector=collector,
        )
        return AttackResult([], record)

    # Each account needs a beneficiary paying into the *next* account in the ring.
    inbound: Dict[str, str] = {}
    for account in accounts:
        inbound[account.account_id] = _beneficiary_for(
            ctx, account.account_id, window.start
        )
        collector.add("beneficiary", inbound[account.account_id])

    # ``cycle_length`` stays exact — it is the structural property the split
    # protocol escalates (3 -> 4 -> 6) — but the number of laundering rounds
    # varies per instance so cycles are not all the same length in rows.
    rounds = vary_count(ctx, params.rounds, spread=0.3)
    n_hops = rounds * len(accounts)
    times = spread_timestamps(ctx, window, n_hops)

    value = params.amount
    for hop in range(n_hops):
        sender = accounts[hop % len(accounts)]
        receiver = accounts[(hop + 1) % len(accounts)]
        device_id, ip_id = account_devices_ips(ctx, sender, times[hop])
        lat, lon = account_home(ctx, sender)
        # Value decays per hop, modelling the cut taken at each step.
        value = round_amount(max(1.0, value * params.decay))
        txn = emit(
            ctx, timestamp=times[hop], account=sender, amount=value,
            device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
            transaction_type="transfer", channel="web",
            beneficiary_id=inbound[receiver.account_id],
        )
        transactions.append(txn)
        collector.add_transaction_entities(txn)

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    severity = severity_for(min(1.0, 0.2 + 0.08 * cycle_length + 0.06 * rounds))
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record)
