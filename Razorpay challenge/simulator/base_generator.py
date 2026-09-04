"""Deterministic generation primitives shared by every generator.

Reproducibility is a Checkpoint 1 exit criterion: *same seed + same config
produces byte-identical output*. That guarantee is fragile if generators share
a single mutable random stream, because then adding one draw anywhere shifts
every subsequent value everywhere. Two mechanisms prevent that:

**Named, independently seeded streams.** :class:`SeedRegistry` derives a child
seed by hashing the master seed together with a stable string path such as
``"entities/devices"`` or ``"attack/fake_merchant_ring/3"``. Streams are
statistically independent and each generator's draws are isolated, so changing
the number of attacks in one split does not perturb entity generation.

**Content-derived IDs.** Identifiers are non-sequential-looking but fully
deterministic: a hash of the stream path plus an index. IDs therefore carry no
ordering information an ML model could exploit — a real leakage risk flagged in
``docs/ML_READINESS.md`` §3 ("sequential IDs" as a giveaway artifact).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, TypeVar

import numpy as np

__all__ = [
    "SeedRegistry",
    "IdMinter",
    "clamp",
    "haversine_km",
    "offset_coordinates",
    "round_amount",
    "seasonal_weight",
    "sample_seasonal_timestamps",
    "weighted_choice_without_replacement",
]

T = TypeVar("T")

_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford-like, no i/l/o/u


class SeedRegistry:
    """Derives independent, reproducible RNG streams from one master seed.

    Streams are addressed by a stable string path. The same path always yields
    the same stream for a given master seed, regardless of the order in which
    streams are requested — that order-independence is what keeps the dataset
    stable as generators evolve.
    """

    def __init__(self, master_seed: int) -> None:
        self._master_seed = int(master_seed)
        self._issued: Dict[str, int] = {}

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def derive_seed(self, path: str) -> int:
        """Return a deterministic 63-bit seed for ``path``."""
        digest = hashlib.sha256(f"{self._master_seed}:{path}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") >> 1
        self._issued[path] = seed
        return seed

    def rng(self, path: str) -> np.random.Generator:
        """Return an independent :class:`numpy.random.Generator` for ``path``."""
        return np.random.default_rng(self.derive_seed(path))

    def issued_paths(self) -> Dict[str, int]:
        """Paths issued so far — recorded in the manifest for traceability."""
        return dict(self._issued)


class IdMinter:
    """Mints stable, non-sequential-looking, prefixed entity identifiers.

    ``mint("cust", 0)`` is deterministic but bears no visible relation to
    ``mint("cust", 1)``, so ID ordering cannot leak creation order or fraud
    status into a model.
    """

    def __init__(self, master_seed: int, namespace: str = "ars") -> None:
        self._master_seed = int(master_seed)
        self._namespace = namespace

    def mint(self, prefix: str, index: int, *, length: int = 12) -> str:
        digest = hashlib.blake2b(
            f"{self._namespace}:{self._master_seed}:{prefix}:{index}".encode("utf-8"),
            digest_size=16,
        ).digest()
        value = int.from_bytes(digest, "big")
        chars: List[str] = []
        for _ in range(length):
            value, rem = divmod(value, len(_ID_ALPHABET))
            chars.append(_ID_ALPHABET[rem])
        return f"{prefix}_{''.join(chars)}"


# --------------------------------------------------------------------------
# Numeric and geographic helpers
# --------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def round_amount(amount: float, *, minimum: float = 0.01) -> float:
    """Round to two decimals with a positive floor.

    Deliberately *not* rounded to whole units: uniformly round fraud amounts
    would be a giveaway artifact (``docs/ML_READINESS.md`` §3).
    """
    return max(minimum, float(round(amount + 1e-12, 2)))


_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def offset_coordinates(
    lat: float, lon: float, distance_km: float, bearing_rad: float
) -> tuple[float, float]:
    """Point ``distance_km`` from (lat, lon) along ``bearing_rad``."""
    ang = distance_km / _EARTH_RADIUS_KM
    phi1, lambda1 = math.radians(lat), math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(ang) + math.cos(phi1) * math.sin(ang) * math.cos(bearing_rad)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(bearing_rad) * math.sin(ang) * math.cos(phi1),
        math.cos(ang) - math.sin(phi1) * math.sin(phi2),
    )
    new_lat = clamp(math.degrees(phi2), -89.9, 89.9)
    new_lon = ((math.degrees(lambda2) + 180.0) % 360.0) - 180.0
    return round(new_lat, 6), round(new_lon, 6)


# --------------------------------------------------------------------------
# Temporal helpers — seasonality (PRD FR-1)
# --------------------------------------------------------------------------


def seasonal_weight(
    ts: datetime, hour_weights: Sequence[float], dow_weights: Sequence[float]
) -> float:
    """Relative activity intensity at ``ts`` from hour-of-day and day-of-week."""
    return float(hour_weights[ts.hour] * dow_weights[ts.weekday()])


def sample_seasonal_timestamps(
    rng: np.random.Generator,
    count: int,
    start: datetime,
    end: datetime,
    hour_weights: Sequence[float],
    dow_weights: Sequence[float],
) -> List[datetime]:
    """Sample ``count`` timestamps in ``[start, end)`` following seasonality.

    Uses rejection sampling (thinning) against the seasonal intensity envelope,
    which yields a genuine inhomogeneous arrival pattern rather than uniform
    times with a cosmetic hour label. Results are sorted ascending.
    """
    if count <= 0:
        return []
    span = (end - start).total_seconds()
    if span <= 0:
        raise ValueError("end must be after start")

    peak = max(hour_weights) * max(dow_weights)
    out: List[datetime] = []
    # Bounded attempts: guarantees termination even in pathological configs.
    max_attempts = max(64, count * 64)
    attempts = 0
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        draw = rng.random(size=min(count * 4, 4096))
        for u in draw:
            if len(out) >= count:
                break
            ts = start + timedelta(seconds=float(u) * span)
            w = seasonal_weight(ts, hour_weights, dow_weights)
            if rng.random() < w / peak:
                out.append(ts)
    while len(out) < count:  # pragma: no cover - defensive fallback
        out.append(start + timedelta(seconds=float(rng.random()) * span))
    out.sort()
    return out


def weighted_choice_without_replacement(
    rng: np.random.Generator, items: Sequence[T], weights: Sequence[float], k: int
) -> List[T]:
    """Sample ``k`` distinct items with probability proportional to weights.

    Uses the exponential-sort (Efraimidis-Spirakis) method, which is stable and
    vectorised. If ``k`` exceeds the population, the whole population is
    returned.
    """
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 0:
        return []
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-12)
    keys = rng.exponential(size=n) / w
    order = np.argsort(keys)[:k]
    return [items[int(i)] for i in order]


@dataclass(frozen=True)
class TimeWindow:
    """A half-open time interval ``[start, end)``.

    Half-open by construction, mirroring the point-in-time window convention in
    ``docs/FEATURE_LEAKAGE_POLICY.md`` §1 so that the same boundary semantics
    apply on both the generation and the feature side.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("TimeWindow end must be strictly after start")

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0


def ensure_utc(ts: datetime) -> datetime:
    """Return ``ts`` as a timezone-aware UTC datetime."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def iter_chunks(seq: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield consecutive chunks of ``size`` from ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
