"""Account takeover (Class A — individual anomaly).

A legitimate, established account is seized. The attacker signs in from a new
device and a new IP, in a location far from the account holder's home, and moves
value quickly before the takeover is noticed. Unlike a transaction spike — which
is loud but uses the victim's *own* infrastructure — a takeover's signature is the
sudden change of context: device, IP, and geography all shift at once, on an
account that had been behaving normally for its whole history.

The victim must have real history for the change to be anomalous, so the family
recruits an existing account with prior activity (``min_activity``). A new device
is created for the attacker; the geographic jump is ``geo_jump_km`` from home.
Whether a new device and/or a changed IP appear is governed by the ``new_device``
/ ``new_ip`` parameters, so a partial takeover (new IP but same device, e.g. a
stolen session) is expressible without changing code.

The connecting *address*, unlike the device, is drawn over four realistic origins
(``ip_origin_weights``): the victim's own address, a shared NAT/CGNAT egress
point, an address already serving some other account, or a freshly provisioned
dedicated one. Minting a dedicated address every time made each one serve exactly
one victim, which turned a train-fitted ``ip_account_degree`` of 1 into a direct
label for this family; the geographic, velocity and behavioural signals carry the
family instead.

This and :mod:`transaction_spike` are the two Class A families, and they are kept
deliberately distinct: takeover changes the *context*, spike changes the *rate*.
Collapsing them would erase a real distinction detectors are expected to make.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import List, Optional

import numpy as np

from simulator.attacks.base import (
    AttackContext,
    AttackResult,
    EntityCollector,
    account_devices_ips,
    attack_window,
    build_record,
    emit,
    jittered_amount,
    merchant_sampler,
    pick_accounts,
    sample_channel,
    severity_for,
    spread_timestamps,
    vary_count,
)
from simulator.base_generator import offset_coordinates
from simulator.config import AccountTakeoverParams
from simulator.entities import (
    DEVICE_TYPES,
    IP_TYPES,
    OS_FAMILIES,
    REGIONS_PER_COUNTRY,
    SYNTHETIC_COUNTRIES,
)

FAMILY = "account_takeover"


def _new_device(ctx: AttackContext, params: AccountTakeoverParams) -> str:
    """Mint the attacker's device.

    Age is drawn from ``params.device_age_days``, which spans hours to most of a
    year. The previous fixed 12-hour freshness made "device created just before the
    transaction" a takeover marker on its own — the audit measured
    ``device_age_days_at_txn`` at AUC 0.9844. Attackers reuse aged devices and
    resold handsets, so a broad age range is both the fix and the more faithful
    behaviour, and it keeps the parameter under config control (PRD FR-2).
    """
    rng = ctx.rng
    fingerprint = "fp_" + "".join(
        f"{int(b):02x}" for b in rng.integers(0, 256, size=8)
    )
    lo, hi = params.device_age_days
    device = ctx.eco.add_device(
        created_at=ctx.window.start - timedelta(days=float(rng.uniform(lo, hi))),
        device_type=DEVICE_TYPES[int(rng.integers(0, len(DEVICE_TYPES)))],
        os_family=OS_FAMILIES[int(rng.integers(0, len(OS_FAMILIES)))],
        fingerprint_hash=fingerprint,
        is_shared_legitimate=False,
    )
    return device.device_id


def _new_ip(
    ctx: AttackContext, params: AccountTakeoverParams, lat: float, lon: float
) -> str:
    """Mint the attacker's IP, aged per ``params.ip_age_days``."""
    rng = ctx.rng
    code, _bl, _bo = SYNTHETIC_COUNTRIES[int(rng.integers(0, len(SYNTHETIC_COUNTRIES)))]
    region = REGIONS_PER_COUNTRY[int(rng.integers(0, len(REGIONS_PER_COUNTRY)))]
    # Takeover IPs skew toward datacenter/VPN, but not exclusively.
    ip_type = (
        "datacenter"
        if rng.random() < params.datacenter_ip_prob
        else IP_TYPES[int(rng.integers(0, len(IP_TYPES)))]
    )
    lo, hi = params.ip_age_days
    ip = ctx.eco.add_ip(
        created_at=ctx.window.start - timedelta(days=float(rng.uniform(lo, hi))),
        ip_address=ctx.eco.next_ip_address(),
        country=code,
        region=region,
        lat=lat,
        lon=lon,
        ip_type=ip_type,
        is_shared_legitimate=False,
    )
    return ip.ip_id


def _choose_takeover_ip(
    ctx: AttackContext,
    params: AccountTakeoverParams,
    own_ip: str,
    lat: float,
    lon: float,
) -> str:
    """Pick the address the takeover connects from, over four realistic origins.

    See :attr:`simulator.config.AccountTakeoverParams.ip_origin_weights` for the
    measurements that forced this. In short: minting a dedicated address per
    takeover meant each one served exactly one account, so a train-fitted
    ``ip_account_degree`` of 1 identified the family outright (AUC 0.9820 at 25%
    prevalence, 0.9847 at 1%). The novelty was never the problem — the *uniqueness*
    was. Attackers overwhelmingly arrive over infrastructure that already carries
    other people's traffic.

    ``other_existing`` samples the address from the legitimate population's own
    registrations, and samples it *proportionally to how many accounts hold it*, so
    the resulting degree distribution is drawn from the same law legitimate traffic
    obeys rather than merely overlapping it. Candidates are filtered to addresses
    that predate the window, since a takeover cannot connect from an address that
    does not exist yet.

    Nothing here touches the geographic jump, the velocity burst, the amount
    scaling or the new device, so every intended takeover signal survives intact.
    """
    rng = ctx.rng
    eco = ctx.eco
    start = ctx.window.start

    w = np.asarray(params.ip_origin_weights, dtype=float)
    w = np.clip(w, 0.0, None)
    if not w.any():  # pragma: no cover - defensive
        return _new_ip(ctx, params, lat, lon)

    # Registered addresses, one entry per (account, ip) link, so an address held by
    # many accounts is more likely to be drawn — the same shape normal traffic has.
    ip_created = {i.ip_id: i.created_at for i in eco.ips}
    shared_flag = {i.ip_id: i.is_shared_legitimate for i in eco.ips}
    shared_pool: List[str] = []
    other_pool: List[str] = []
    for account_id, ip_ids in eco.account_ips.items():
        for ip_id in ip_ids:
            if ip_created.get(ip_id, start) > start:
                continue
            if shared_flag.get(ip_id):
                shared_pool.append(ip_id)
            elif ip_id != own_ip:
                other_pool.append(ip_id)

    # Zero out origins with no candidates rather than silently falling through to a
    # fresh address, which would quietly restore the defect.
    if not shared_pool:
        w[1] = 0.0
    if not other_pool:
        w[2] = 0.0
    total = float(w.sum())
    if total <= 0.0:  # pragma: no cover - defensive
        return _new_ip(ctx, params, lat, lon)

    origin = int(rng.choice(4, p=w / total))
    if origin == 0:
        return own_ip
    if origin == 1:
        return shared_pool[int(rng.integers(0, len(shared_pool)))]
    if origin == 2:
        return other_pool[int(rng.integers(0, len(other_pool)))]
    return _new_ip(ctx, params, lat, lon)


def generate(ctx: AttackContext, params: AccountTakeoverParams) -> AttackResult:
    rng = ctx.rng
    window = attack_window(ctx, params.duration_hours)
    collector = EntityCollector()
    transactions: List = []

    candidates = pick_accounts(ctx, 1, min_activity=8)
    if not candidates:  # pragma: no cover - defensive
        record = build_record(
            ctx, family=FAMILY, window=window, severity="low",
            params=asdict(params), transactions=[], collector=collector,
        )
        return AttackResult([], record)
    account = candidates[0]
    customer = ctx.eco.customer_by_id[account.customer_id]

    # Location of the takeover: a jump of ``geo_jump_km`` from the victim's home.
    jump_lat, jump_lon = offset_coordinates(
        customer.home_lat, customer.home_lon,
        params.geo_jump_km, float(rng.random()) * 2 * np.pi,
    )

    # Fall back to the victim's own device/IP for whichever the scenario keeps.
    # Resolved as of the window start, the earliest instant this attack can emit.
    own_device, own_ip = account_devices_ips(ctx, account, window.start)
    device_id = _new_device(ctx, params) if params.new_device else own_device
    ip_id = (
        _choose_takeover_ip(ctx, params, own_ip, jump_lat, jump_lon)
        if params.new_ip
        else own_ip
    )

    # Cash-out volume: ``velocity_multiplier`` days of the victim's usual
    # activity, compressed into the short takeover window.
    baseline_daily = max(0.0, account.baseline_tx_frequency)
    # Jittered for the same reason as the spike family: without it, every takeover
    # of a quiet account is exactly ``min_transactions`` rows.
    n_tx = vary_count(
        ctx,
        max(
            params.min_transactions,
            int(round(baseline_daily * params.velocity_multiplier)),
        ),
        spread=0.2,
    )
    pick_merchant = merchant_sampler(ctx)
    times = spread_timestamps(ctx, window, n_tx)
    mean = account.baseline_amount_mean * params.amount_scale
    std = account.baseline_amount_std * params.amount_scale

    for ts in times:
        # Cash-out skews toward withdrawals/purchases, drawn near the jump point.
        lat, lon = offset_coordinates(
            jump_lat, jump_lon, float(rng.random()) * 12.0, float(rng.random()) * 2 * np.pi
        )
        r = rng.random()
        tx_type = "withdrawal" if r < 0.35 else "purchase"
        merchant_id: Optional[str] = None if tx_type == "withdrawal" else pick_merchant(ts)
        txn = emit(
            ctx, timestamp=ts, account=account, amount=jittered_amount(ctx, mean, std),
            device_id=device_id, ip_id=ip_id, lat=lat, lon=lon,
            transaction_type=tx_type, channel=sample_channel(ctx),
            merchant_id=merchant_id,
        )
        transactions.append(txn)
        collector.add_transaction_entities(txn)

    transactions.sort(key=lambda t: (t.timestamp, t.transaction_id))
    severity = severity_for(
        min(1.0, 0.3 + params.velocity_multiplier / 12.0 + params.geo_jump_km / 8000.0)
    )
    record = build_record(
        ctx, family=FAMILY, window=window, severity=severity,
        params=asdict(params), transactions=transactions, collector=collector,
    )
    return AttackResult(transactions, record)
