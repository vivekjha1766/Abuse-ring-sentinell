"""Shared device ring (Class B — coordinated abuse).

Many accounts operate from a small pool of device fingerprints — a bot farm or
a mule crew working from a handful of physical machines. The coordinated signal
is device concentration: far more accounts per device than a household explains,
combined with those accounts transacting in a tight window.

Why this is hard, and why the ring must be *embedded* (``docs/ML_READINESS.md``
§8): the ecosystem already contains legitimate device sharing — households and
shared machines flagged ``is_shared_legitimate``. So "several accounts on one
device" is ordinary. The ring is distinguished only by the *degree* of sharing
and its temporal coordination, not by the mere existence of a shared device. The
member accounts are existing legitimate accounts that keep their normal traffic
on their own devices; only their *ring* activity runs through the shared pool.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import List

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
    jittered_amount,
    merchant_sampler,
    pick_accounts,
    sample_channel,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.config import SharedDeviceRingParams
from simulator.entities import DEVICE_TYPES, OS_FAMILIES

FAMILY = "shared_device_ring"

#: Ring-controlled devices may not exceed this share of the legitimate device
#: population. Benign household sharing must stay visible among multi-account
#: devices, or "this device serves several accounts" becomes a fraud oracle.
MAX_RING_DEVICE_SHARE = 0.12


def _create_ring_device(ctx: AttackContext):
    """A device controlled by the ring.

    Crucially *not* flagged ``is_shared_legitimate`` — that flag is ground truth
    for benign sharing. Ring devices are created without it, but are otherwise
    indistinguishable in their attributes from ordinary devices.
    """
    rng = ctx.rng
    fingerprint = "fp_" + "".join(
        f"{int(b):02x}" for b in rng.integers(0, 256, size=8)
    )
    return ctx.eco.add_device(
        created_at=ctx.window.start - timedelta(days=float(rng.random()) * 30.0 + 1.0),
        device_type=DEVICE_TYPES[int(rng.integers(0, len(DEVICE_TYPES)))],
        os_family=OS_FAMILIES[int(rng.integers(0, len(OS_FAMILIES)))],
        fingerprint_hash=fingerprint,
        is_shared_legitimate=False,
    )


def generate(ctx: AttackContext, params: SharedDeviceRingParams) -> AttackResult:
    rng = ctx.rng
    window = attack_window(ctx, params.duration_hours)
    collector = EntityCollector()
    transactions: List = []

    # Mint fresh ring devices while the device population can absorb them, then top
    # up by recycling devices from earlier rings. Without the cap, a high-prevalence
    # configuration creates so many ring devices that they swamp the benign
    # household-sharing population — measured at 25% prevalence, only 0.1% of
    # multi-account devices were fraud-free, which turns "shared device" into a
    # near-perfect rule and defeats the whole point of seeding benign sharing
    # (``docs/ML_READINESS.md`` §3). Device farms genuinely do reuse handsets across
    # rings, so recycling is also the more faithful behaviour.
    legit_devices = ctx.eco.config.ecosystem.n_devices
    ring_pool = ctx.eco.devices[legit_devices:]
    want_devices = max(1, params.n_devices)
    n_new = capped_entities(
        ctx,
        requested=want_devices,
        existing=len(ring_pool),
        population=legit_devices,
        max_share=MAX_RING_DEVICE_SHARE,
    )
    ring_devices = [_create_ring_device(ctx) for _ in range(n_new)]
    recyclable = list(ring_pool)
    while len(ring_devices) < want_devices and recyclable:
        ring_devices.append(recyclable.pop(int(rng.integers(0, len(recyclable)))))
    if not ring_devices:  # pragma: no cover - defensive
        ring_devices = [_create_ring_device(ctx)]
    for device in ring_devices:
        collector.add("device", device.device_id)

    accounts = pick_accounts(ctx, vary_count(ctx, params.n_accounts, spread=0.2))
    if not accounts:  # pragma: no cover - defensive
        record = build_record(
            ctx, family=FAMILY, window=window, severity="low",
            params=asdict(params), transactions=[], collector=collector,
        )
        return AttackResult([], record)

    # Ring purchases go to *real* merchants, sampled with the same popularity
    # weighting as normal traffic — the ring's signature is the shared device.
    pick_merchant = merchant_sampler(ctx)

    for account in accounts:
        # Per-account, so ring members are not clones of one another.
        n_tx = vary_count(ctx, params.tx_per_account, spread=0.3)
        times = spread_timestamps(ctx, window, n_tx)
        # The account's own device/IP for the non-ring share of activity, so the
        # ring is not a component wholly separate from the account's real one.
        own_device, own_ip = account_devices_ips(ctx, account, window.start)
        for ts in times:
            use_ring = rng.random() < params.sharing_density
            device_id = (
                ring_devices[int(rng.integers(0, len(ring_devices)))].device_id
                if use_ring
                else own_device
            )
            amount = jittered_amount(ctx, params.amount_mean, params.amount_std)
            lat, lon = account_home(ctx, account)
            txn = emit(
                ctx, timestamp=ts, account=account, amount=amount,
                device_id=device_id, ip_id=own_ip, lat=lat, lon=lon,
                transaction_type="purchase", channel=sample_channel(ctx),
                merchant_id=pick_merchant(ts),
            )
            transactions.append(txn)
            collector.add_transaction_entities(txn)

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    concentration = len(accounts) / max(1, len(ring_devices))
    severity = severity_for(min(1.0, 0.15 + 0.03 * concentration + 0.2 * params.sharing_density))
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record)
