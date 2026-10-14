"""Synthetic identity cluster (Class B — coordinated abuse).

A batch of fabricated identities created close together in time, sharing more
attributes than chance would produce (the same country, similar spending
profile), then ramping up ordinary-looking activity to build history before the
inevitable bust-out. Unlike the other families, these identities genuinely *are*
new entities — that is the definition of synthetic identity fraud — so the
challenge is to create them without letting "newness" become a trivial label.

Two embedding decisions (``docs/ML_READINESS.md`` §8) address that:

* The synthetic accounts transact with **existing** merchants using normal
  popularity weighting, and route through **existing** shared IP pools, so they
  are connected to the same infrastructure legitimate accounts use rather than
  forming an isolated island.
* The cluster is created during the split window but its members' creation
  timestamps are jittered across ``creation_window_hours`` and their first
  activity is spread over ``ramp_days``, so they do not all appear at one instant.

``attribute_correlation`` controls how much cluster members resemble each other:
at high values they share a country and spending profile; at low values they are
more heterogeneous. That single parameter therefore tunes how detectable the
cluster is, with no shape hardcoded.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import List

import numpy as np

from simulator.attacks.base import (
    AttackContext,
    AttackResult,
    EntityCollector,
    attack_window,
    build_record,
    capped_entities,
    emit,
    emit_cover,
    jittered_amount,
    merchant_sampler,
    sample_channel,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.base_generator import TimeWindow, offset_coordinates
from simulator.config import SyntheticIdentityParams
from simulator.entities import (
    ACCOUNT_TYPES,
    CUSTOMER_SEGMENTS,
    REGIONS_PER_COUNTRY,
    RISK_BANDS,
    SEGMENT_TX_FREQUENCY,
    SPENDING_PROFILES,
    SYNTHETIC_COUNTRIES,
)

FAMILY = "synthetic_identity_cluster"

#: Fabricated customers may not exceed this share of the real customer population.
#: See :func:`simulator.attacks.base.capped_entities` — at high prevalence the
#: uncapped behaviour let synthetic identities reach a third of all customers,
#: which distorts every customer-level base rate in the dataset.
MAX_SYNTHETIC_CUSTOMER_SHARE = 0.20


def _create_identity(ctx: AttackContext, params: SyntheticIdentityParams, shared: dict):
    """Fabricate one customer + account, correlated with the cluster template.

    With probability ``attribute_correlation`` a field is copied from the shared
    template; otherwise it is drawn independently. So higher correlation ⇒ a
    tighter, more detectable cluster, entirely under parameter control.
    """
    rng = ctx.rng

    def correlated(shared_value, independent):
        return shared_value if rng.random() < params.attribute_correlation else independent()

    country_idx = correlated(
        shared["country_idx"], lambda: int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))
    )
    code, base_lat, base_lon = SYNTHETIC_COUNTRIES[country_idx]
    region = correlated(
        shared["region"],
        lambda: REGIONS_PER_COUNTRY[int(rng.integers(0, len(REGIONS_PER_COUNTRY)))],
    )
    lat, lon = offset_coordinates(
        base_lat, base_lon, float(rng.random()) * 40.0, float(rng.random()) * 2 * np.pi
    )
    profile = correlated(
        shared["profile"],
        lambda: SPENDING_PROFILES[int(rng.integers(0, len(SPENDING_PROFILES)))],
    )
    segment = correlated(
        shared["segment"],
        lambda: CUSTOMER_SEGMENTS[int(rng.integers(0, len(CUSTOMER_SEGMENTS)))],
    )
    # The batch is fabricated *before* the bust-out window and jittered around a
    # shared anchor, so members are created close to one another (the cluster
    # signal) without being young at the moment they transact (the former oracle).
    anchor = shared["anchor"]
    created = anchor + timedelta(hours=float(rng.random()) * params.creation_window_hours)
    created = min(created, shared["window"].start - timedelta(hours=1.0))
    customer = ctx.eco.add_customer(
        created_at=created,
        country=code,
        region=region,
        home_lat=lat,
        home_lon=lon,
        segment=segment,
        spending_profile=profile,
        risk_band=RISK_BANDS[int(rng.integers(0, len(RISK_BANDS)))],
        is_synthetic_identity=True,
    )
    amount_mean = max(10.0, float(rng.normal(params.amount_mean, params.amount_mean * 0.2)))
    # Activity drawn from the *same* segment law legitimate accounts use, so a
    # fabricated identity is not detectable by being unusually quiet. See
    # :data:`simulator.entities.SEGMENT_TX_FREQUENCY`.
    f_lo, f_hi = SEGMENT_TX_FREQUENCY[segment]
    account = ctx.eco.add_account(
        customer_id=customer.customer_id,
        created_at=created,
        account_type=ACCOUNT_TYPES[int(rng.integers(0, len(ACCOUNT_TYPES)))],
        status="active",
        baseline_tx_frequency=round(float(rng.uniform(f_lo, f_hi)), 4),
        baseline_amount_mean=round(amount_mean, 2),
        baseline_amount_std=round(amount_mean * 0.6, 2),
    )
    # Wire the new account into existing shared infrastructure, so it is not an
    # isolated node: reuse an existing shared IP and an existing device. Both must
    # predate the identity itself, because the incubation history below is emitted
    # from ``account.created_at`` onward — a device acquired later would give those
    # rows a negative device age.
    shared_ips = [
        i.ip_id
        for i in ctx.eco.ips
        if i.is_shared_legitimate and i.created_at <= created
    ]
    if not shared_ips:
        shared_ips = [i.ip_id for i in ctx.eco.ips if i.created_at <= created]
    if not shared_ips:  # pragma: no cover - defensive
        shared_ips = [min(ctx.eco.ips, key=lambda i: i.created_at).ip_id]
    eligible_devices = [d.device_id for d in ctx.eco.devices if d.created_at <= created]
    if not eligible_devices:  # pragma: no cover - defensive
        eligible_devices = [min(ctx.eco.devices, key=lambda d: d.created_at).device_id]
    device_id = eligible_devices[int(rng.integers(0, len(eligible_devices)))]
    ip_id = shared_ips[int(rng.integers(0, len(shared_ips)))]
    ctx.eco.account_devices.setdefault(account.account_id, []).append(device_id)
    ctx.eco.account_ips.setdefault(account.account_id, []).append(ip_id)
    return customer, account, device_id, ip_id


def _recyclable_identities(ctx: AttackContext, exclude: set) -> List[tuple]:
    """Previously fabricated identities, in the same shape ``_create_identity`` returns.

    Lets a cluster reuse synthetic people once the population cap binds. Returns
    ``(customer, account, device_id, ip_id)`` tuples so callers cannot tell a
    recycled member from a fresh one.
    """
    out: List[tuple] = []
    for account in ctx.eco.accounts:
        customer = ctx.eco.customer_by_id.get(account.customer_id)
        if customer is None or not customer.is_synthetic_identity:
            continue
        if customer.customer_id in exclude:
            continue
        devices = ctx.eco.account_devices.get(account.account_id)
        ips = ctx.eco.account_ips.get(account.account_id)
        if not devices or not ips:
            continue
        out.append((customer, account, devices[0], ips[0]))
    return out


def generate(ctx: AttackContext, params: SyntheticIdentityParams) -> AttackResult:
    rng = ctx.rng
    window = attack_window(ctx, max(params.ramp_days * 24.0, params.creation_window_hours))
    collector = EntityCollector()
    transactions: List = []

    # One shared template the cluster is correlated toward.
    inc_lo, inc_hi = params.incubation_days
    shared = {
        "country_idx": int(rng.integers(0, len(SYNTHETIC_COUNTRIES))),
        "region": REGIONS_PER_COUNTRY[int(rng.integers(0, len(REGIONS_PER_COUNTRY)))],
        "profile": SPENDING_PROFILES[int(rng.integers(0, len(SPENDING_PROFILES)))],
        "segment": CUSTOMER_SEGMENTS[int(rng.integers(0, len(CUSTOMER_SEGMENTS)))],
        "window": window,
        # Batch fabrication instant, one incubation period before the bust-out.
        "anchor": window.start - timedelta(days=float(rng.uniform(inc_lo, inc_hi))),
    }

    # Fabricate new identities while the customer population can absorb them, then
    # top up the cluster by recruiting identities fabricated by earlier clusters.
    # Identity farms genuinely do resell the same fabricated people across rings, so
    # the overlap is realistic — and it stops the synthetic population from growing
    # without bound at high prevalence.
    want = vary_count(ctx, params.n_identities, spread=0.2)
    n_new = capped_entities(
        ctx,
        requested=want,
        existing=sum(1 for c in ctx.eco.customers if c.is_synthetic_identity),
        population=sum(1 for c in ctx.eco.customers if not c.is_synthetic_identity),
        max_share=MAX_SYNTHETIC_CUSTOMER_SHARE,
    )
    members = [_create_identity(ctx, params, shared) for _ in range(n_new)]
    recyclable = _recyclable_identities(ctx, exclude={m[0].customer_id for m in members})
    while len(members) < want and recyclable:
        members.append(recyclable.pop(int(rng.integers(0, len(recyclable)))))
    if not members:  # pragma: no cover - defensive
        members = [_create_identity(ctx, params, shared)]
    pick_merchant = merchant_sampler(ctx)
    cover: List = []
    observable_from = ctx.eco.config.warmup_start

    for customer, account, device_id, ip_id in members:
        collector.add("customer", customer.customer_id)
        collector.add("account", account.account_id)
        collector.add("device", device_id)
        collector.add("ip", ip_id)

        # --- incubation: ordinary-looking history, legitimately labelled ------
        # The identity was fabricated before the window and nursed until it looked
        # established. We can only observe the part of that incubation that falls
        # inside the dataset, so emission starts at the later of creation and the
        # warm-up boundary. Without this history a fabricated identity would be an
        # aged account with no transactions at all, which merely trades the
        # "suspiciously young" shortcut for a "suspiciously empty" one.
        hist_start = max(account.created_at, observable_from)
        if hist_start < window.start:
            hist_days = (window.start - hist_start).total_seconds() / 86400.0
            n_hist = int(
                ctx.rng.poisson(
                    max(
                        0.0,
                        account.baseline_tx_frequency
                        * hist_days
                        * params.history_activity_scale,
                    )
                )
            )
            if n_hist > 0:
                for ts in spread_timestamps(
                    ctx, TimeWindow(hist_start, window.start), n_hist, jitter=0.6
                ):
                    lat, lon = offset_coordinates(
                        customer.home_lat, customer.home_lon,
                        float(rng.random()) * 6.0, float(rng.random()) * 2 * np.pi,
                    )
                    cover.append(
                        emit_cover(
                            ctx, timestamp=ts, account=account,
                            amount=jittered_amount(
                                ctx, account.baseline_amount_mean,
                                account.baseline_amount_std,
                            ),
                            device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
                            transaction_type="purchase", channel=sample_channel(ctx),
                            merchant_id=pick_merchant(ts),
                        )
                    )

        # --- bust-out: the fraudulent activity, contained in the split window --
        n_tx = vary_count(ctx, params.tx_per_identity, spread=0.3)
        member_start = max(window.start, customer.created_at, account.created_at)
        if member_start >= window.end:  # pragma: no cover - defensive
            continue
        # Volume follows rate x lifetime, the same law legitimate accounts obey, so
        # the identity accumulates a history proportionate to how long it has
        # existed. ``tx_per_identity`` becomes a floor for the deliberate bust-out
        # burst rather than the entire footprint.
        live_days = (window.end - member_start).total_seconds() / 86400.0
        organic = int(ctx.rng.poisson(max(0.0, account.baseline_tx_frequency * live_days)))
        n_tx = max(n_tx, organic)
        times = spread_timestamps(
            ctx, TimeWindow(member_start, window.end), n_tx, jitter=0.5
        )
        for ts in times:
            amount = jittered_amount(
                ctx, account.baseline_amount_mean, account.baseline_amount_std
            )
            lat, lon = offset_coordinates(
                customer.home_lat, customer.home_lon,
                float(rng.random()) * 6.0, float(rng.random()) * 2 * np.pi,
            )
            txn = emit(
                ctx, timestamp=ts, account=account, amount=amount,
                device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
                transaction_type="purchase", channel=sample_channel(ctx),
                merchant_id=pick_merchant(ts),
            )
            transactions.append(txn)
            collector.add_transaction_entities(txn)

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    severity = severity_for(
        min(1.0, 0.15 + 0.04 * params.n_identities + 0.2 * params.attribute_correlation)
    )
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record, cover_transactions=cover)
