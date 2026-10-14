"""Fake merchant ring (Class B — coordinated abuse).

A cluster of newly onboarded merchants that exist only to collect payments,
which are then funnelled to a small set of shared beneficiaries. The give-away is
*not* any single transaction — each looks like an ordinary purchase — but the
coordinated structure: several fresh merchants, an overlapping set of paying
accounts, and value converging on shared payout beneficiaries.

Embedding (``docs/ML_READINESS.md`` §8): the paying accounts are **existing**
legitimate accounts recruited into the ring, so they still carry their normal
traffic to normal merchants. The fake merchants are genuinely new — that is what
merchant onboarding fraud *is* — but they are paid by real accounts, which keeps
them attached to the main graph instead of forming an island.

Modelling note: merchants are not account holders in this schema
(``DATA_DICTIONARY.md`` §2), so the payout leg is represented as transfers from
ring accounts to the ring's beneficiaries rather than from the merchant itself.
"""

from __future__ import annotations

from dataclasses import asdict
from bisect import bisect_right
from typing import Dict, List

import numpy as np

from simulator.attacks.base import (
    AttackContext,
    AttackResult,
    EntityCollector,
    account_devices_ips,
    account_home,
    attack_window,
    build_record,
    capped_entities,
    emit,
    emit_cover,
    entity_created_at,
    jittered_amount,
    pick_accounts,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.base_generator import TimeWindow, offset_coordinates, round_amount
from simulator.config import FakeMerchantRingParams
from simulator.entities import (
    BENEFICIARY_TYPES,
    MERCHANT_CATEGORIES,
    SYNTHETIC_COUNTRIES,
)

FAMILY = "fake_merchant_ring"

#: Fake merchants may not exceed this share of the legitimate merchant population.
#: Without a cap, a high-prevalence configuration injects enough rings that fake
#: merchants become the *majority* of the merchant population, which makes their
#: detection a base-rate artefact rather than a behavioural one.
MAX_FAKE_MERCHANT_SHARE = 0.35

#: Payout beneficiaries the rings may create, as a share of the legitimate payee
#: population. Much smaller than the merchant share because the payee population is
#: itself much larger: it is sized so accounts hold short *private* payee lists
#: (``EcosystemConfig.beneficiaries_per_account``), which puts ~1500 beneficiaries in
#: the ecosystem. A proportional cap would license hundreds of fresh payout
#: beneficiaries, and every one of those is an entity no TRAIN row has ever seen —
#: which converts a merchant-ring cap into a ``beneficiary_unseen_in_train`` novelty
#: leak. 0.08 keeps the absolute ceiling near where it was under the old 260-payee
#: population, so rings still recycle payout beneficiaries across instances.
MAX_RING_BENEFICIARY_SHARE = 0.08


def _create_fake_merchant(ctx: AttackContext, params: FakeMerchantRingParams):
    """Onboard one collusive merchant.

    Its popularity weight is drawn from the same range the legitimate *small*
    tier uses, so the ring cannot be found by merchant weight alone, and its
    onboarding date comes from the same law legitimate merchants obey
    (:func:`simulator.attacks.base.entity_created_at`) rather than a narrow
    "1-20 days before the split" window that would make shells identifiable by age.
    """
    rng = ctx.rng
    code, base_lat, base_lon = SYNTHETIC_COUNTRIES[
        int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
    ]
    lat, lon = offset_coordinates(
        base_lat, base_lon, float(rng.random()) * 90.0, float(rng.random()) * 2 * np.pi
    )
    created = entity_created_at(
        ctx, not_after=ctx.window.start, history_days=900.0
    )
    ticket = max(5.0, float(rng.normal(params.amount_mean, params.amount_std * 0.5)))
    return ctx.eco.add_merchant(
        weight=float(rng.uniform(0.4, 3.0)),
        created_at=created,
        category=MERCHANT_CATEGORIES[int(rng.integers(0, len(MERCHANT_CATEGORIES)))],
        country=code,
        lat=lat,
        lon=lon,
        size_tier="small",
        avg_ticket_size=round(ticket, 2),
        is_fake=True,
    )


def _create_payout_beneficiary(ctx: AttackContext):
    rng = ctx.rng
    code, _lat, _lon = SYNTHETIC_COUNTRIES[
        int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
    ]
    return ctx.eco.add_beneficiary(
        created_at=entity_created_at(
            ctx, not_after=ctx.window.start, history_days=500.0
        ),
        beneficiary_type=BENEFICIARY_TYPES[int(rng.integers(0, len(BENEFICIARY_TYPES)))],
        country=code,
        linked_account_id=None,
    )


def generate(ctx: AttackContext, params: FakeMerchantRingParams) -> AttackResult:
    rng = ctx.rng
    window = attack_window(ctx, params.duration_hours)
    collector = EntityCollector()
    transactions: List = []

    # Recruit from the pre-created shell pool first. Those storefronts were built
    # alongside the legitimate population, so the normal-traffic sampler has already
    # given them an organic trading history — which is what a real shell buys when it
    # passes acquirer review. Onboarding a shell *here* instead would produce a
    # merchant whose every row is fraudulent, because normal traffic has already been
    # generated by the time attacks run; that path is kept only for configurations
    # with no pool. Eligibility is by creation time, since a shell that joins later
    # in the window cannot be paid at the start of it.
    want_merchants = max(1, params.n_merchants)
    pool = [
        m
        for m in ctx.eco.merchants
        if m.is_fake and m.created_at <= window.start
    ]
    merchants = []
    while len(merchants) < want_merchants and pool:
        merchants.append(pool.pop(int(rng.integers(0, len(pool)))))
    if len(merchants) < want_merchants:
        n_new = capped_entities(
            ctx,
            requested=want_merchants - len(merchants),
            existing=sum(1 for m in ctx.eco.merchants if m.is_fake),
            population=sum(1 for m in ctx.eco.merchants if not m.is_fake),
            max_share=MAX_FAKE_MERCHANT_SHARE,
        )
        merchants.extend(_create_fake_merchant(ctx, params) for _ in range(n_new))
    if not merchants:  # pragma: no cover - only if no merchant is usable at all
        merchants = [_create_fake_merchant(ctx, params)]

    # Beneficiaries carry no ``is_fake`` flag, but they are appended in creation
    # order, so everything past the configured baseline is attack-created.
    legit_beneficiaries = ctx.eco.config.ecosystem.n_beneficiaries
    ring_pool = ctx.eco.beneficiaries[legit_beneficiaries:]
    want_beneficiaries = max(1, params.n_beneficiaries)
    n_new_ben = capped_entities(
        ctx,
        requested=want_beneficiaries,
        existing=len(ring_pool),
        population=legit_beneficiaries,
        max_share=MAX_RING_BENEFICIARY_SHARE,
    )
    beneficiaries = [_create_payout_beneficiary(ctx) for _ in range(n_new_ben)]
    # Recycling is restricted twice over. Only *unlinked* payout beneficiaries are
    # eligible, because a beneficiary carrying a ``linked_account_id`` belongs to a
    # circular-flow cycle and pays into a ring account rather than out to a shell
    # operator. And only those that already existed when this window opened, since
    # a cycle registered later in the same split would otherwise be paid before it
    # was created (measured: 9 rows at -35.7 days). ``existing`` above deliberately
    # still counts the whole attack-created pool, so the population cap keeps its
    # original meaning.
    recyclable_ben = [
        b
        for b in ring_pool
        if b.linked_account_id is None and b.created_at <= window.start
    ]
    while len(beneficiaries) < want_beneficiaries and recyclable_ben:
        beneficiaries.append(
            recyclable_ben.pop(int(rng.integers(0, len(recyclable_ben))))
        )
    if not beneficiaries:  # pragma: no cover - defensive
        beneficiaries = [_create_payout_beneficiary(ctx)]
    for merchant in merchants:
        collector.add("merchant", merchant.merchant_id)
    for beneficiary in beneficiaries:
        collector.add("beneficiary", beneficiary.beneficiary_id)

    accounts = pick_accounts(ctx, vary_count(ctx, params.n_accounts, spread=0.2))
    if not accounts:  # pragma: no cover - defensive
        record = build_record(
            ctx, family=FAMILY, window=window, severity="low",
            params=asdict(params), transactions=[], collector=collector,
        )
        return AttackResult([], record)

    # Volume derives from the ring's own size and intensity: each account works
    # roughly once per ring merchant, scaled by intensity.
    per_account = vary_count(
        ctx, max(1, int(round(len(merchants) * params.intensity))), spread=0.25
    )
    purchase_times = spread_timestamps(ctx, window, len(accounts) * per_account)

    collected = 0.0
    cursor = 0
    for account in accounts:
        device_id, ip_id = account_devices_ips(ctx, account, window.start)
        for _ in range(per_account):
            if cursor >= len(purchase_times):
                break
            merchant = merchants[int(rng.integers(0, len(merchants)))]
            amount = jittered_amount(ctx, params.amount_mean, params.amount_std)
            collected += amount
            lat, lon = account_home(ctx, account)
            txn = emit(
                ctx, timestamp=purchase_times[cursor], account=account, amount=amount,
                device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
                transaction_type="purchase", merchant_id=merchant.merchant_id,
            )
            cursor += 1
            transactions.append(txn)
            collector.add_transaction_entities(txn)

    # Payout leg: ``payout_ratio`` of collected value forwarded to the ring's
    # beneficiaries, over as many transfers as intensity implies.
    payout_total = collected * params.payout_ratio
    if payout_total > 0:
        n_payouts = max(
            len(beneficiaries), int(round(len(beneficiaries) * params.intensity))
        )
        payout_times = spread_timestamps(ctx, window, n_payouts)
        per_payout = payout_total / n_payouts
        for k in range(n_payouts):
            account = accounts[int(rng.integers(0, len(accounts)))]
            beneficiary = beneficiaries[k % len(beneficiaries)]
            device_id, ip_id = account_devices_ips(ctx, account, payout_times[k])
            amount = round_amount(
                max(1.0, float(rng.normal(per_payout, per_payout * 0.15)))
            )
            lat, lon = account_home(ctx, account)
            txn = emit(
                ctx, timestamp=payout_times[k], account=account, amount=amount,
                device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
                transaction_type="transfer", channel="web",
                beneficiary_id=beneficiary.beneficiary_id,
            )
            transactions.append(txn)
            collector.add_transaction_entities(txn)

    # --- cover traffic: shortfall top-up only ----------------------------
    # A shell whose history is entirely fraudulent is a data defect. Measured: with
    # no clean history, fake merchants sat at a 0.50-0.76 train fraud rate against
    # 0.008-0.118 for legitimate merchants, only 1.5% of which exceeded 0.27 — so a
    # single threshold on a train-fitted per-merchant fraud rate separated the ring at
    # AUC 0.9830, and 100% of held-out ring rows sat on shells already active in
    # TRAIN, so the encoding transferred perfectly.
    #
    # The pool fixes that structurally: a shell's clean rows are organic, drawn from
    # the normal-traffic budget. This pass only covers the residual case where a
    # shell's organic volume in *this split* is too thin to keep its accumulated ring
    # rows under ``max_fraud_share``. When organic volume already suffices — the
    # common case — nothing is injected and the dataset pays nothing. Confining the
    # top-up to the split window matters too: the earlier lifetime-spanning version
    # pushed cover backwards into TRAIN and skewed the split fractions to
    # 0.683/0.167/0.150 against a configured 0.600/0.200/0.200.
    cover: List = []
    ring_rows_per_merchant: Dict[str, int] = {}
    for txn in transactions:
        if txn.merchant_id is not None:
            ring_rows_per_merchant[txn.merchant_id] = (
                ring_rows_per_merchant.get(txn.merchant_id, 0) + 1
            )
    share = min(0.95, max(0.01, params.max_fraud_share))
    clean_per_fraud = (1.0 - share) / share

    shortfalls: Dict[str, int] = {}
    for merchant_id, fraud_rows in ring_rows_per_merchant.items():
        # Accumulated exposure, so recycling by a later ring cannot push the share
        # back up: this instance's rows are added to the split-wide tally first.
        total_fraud = ctx.merchant_fraud_rows.get(merchant_id, 0) + fraud_rows
        ctx.merchant_fraud_rows[merchant_id] = total_fraud
        clean = ctx.merchant_activity.get(merchant_id, 0)
        needed = int(round(total_fraud * clean_per_fraud))
        if needed > clean:
            shortfalls[merchant_id] = needed - clean

    if shortfalls:
        outsiders = [
            a
            for a in pick_accounts(ctx, min(len(accounts) * 4 + 8, 64))
            if a.account_id not in {acc.account_id for acc in accounts}
        ]
        # Buyer eligibility is per timestamp, not per attack: ``pick_accounts`` only
        # guarantees existence at the window start, and cover rows are spread across
        # the window. Sorting once lets each row draw from the eligible prefix in
        # log time. Without this, cover landed on accounts not yet opened (measured:
        # 519 rows at -109.1 days).
        outsiders.sort(key=lambda a: a.created_at)
        outsider_created = [a.created_at for a in outsiders]
        for merchant_id, n_cover in sorted(shortfalls.items()):
            if not outsiders:  # pragma: no cover - defensive
                break
            merchant = ctx.eco.merchant_by_id[merchant_id]
            life_start = max(merchant.created_at, window.start)
            if life_start >= window.end:  # pragma: no cover - defensive
                continue
            for ts in spread_timestamps(
                ctx, TimeWindow(life_start, window.end), n_cover, jitter=0.6
            ):
                eligible = bisect_right(outsider_created, ts)
                if eligible == 0:
                    continue
                buyer = outsiders[int(rng.integers(0, eligible))]
                device_id, ip_id = account_devices_ips(ctx, buyer, ts)
                lat, lon = account_home(ctx, buyer)
                cover.append(
                    emit_cover(
                        ctx, timestamp=ts, account=buyer,
                        amount=jittered_amount(
                            ctx, buyer.baseline_amount_mean, buyer.baseline_amount_std
                        ),
                        device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
                        transaction_type="purchase", merchant_id=merchant_id,
                    )
                )
            # Later instances recycling this shell must see what was just added.
            ctx.merchant_activity[merchant_id] = (
                ctx.merchant_activity.get(merchant_id, 0) + n_cover
            )

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    severity = severity_for(
        min(1.0, 0.2 + 0.05 * len(merchants) + 0.02 * len(accounts) + 0.2 * params.intensity)
    )
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record, cover_transactions=cover)
