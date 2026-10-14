"""The feature safety contract: which fields may be read, and how.

``simulator/leakage.py`` already answers one question — *which fields are the
answer key* — and this module deliberately does not restate it. ``LABEL_FIELDS``
and ``ENTITY_LABEL_FIELDS`` are imported from there, and :data:`RATIONALE` records
why the remaining fields are classified as they are.

The contract exists because "don't leak the future" is not actionable. A field is
unsafe for one of several different reasons, and the fix differs in each case:

``LABEL``
    Ground truth. Evaluation only. Owned by ``simulator.leakage``.
``IDENTIFIER``
    A join key. Legitimate for grouping history, never a model input: an ID
    carries no signal a real system could act on, and lexical ID order is a
    generator artifact (the Checkpoint 1 audit probes
    ``transaction_id_lexical_value`` for exactly this reason).
``POINT_IN_TIME_SAFE``
    Observable at authorization time for the row being scored. Readable directly.
``AGGREGATE_ONLY``
    Observable for *earlier* rows but not for the row being scored. Reachable only
    through :mod:`ml.point_in_time`, never off the row itself.
``FORBIDDEN``
    A generator parameter or an untimestamped mutable state field. No real system
    could read it as-of ``T``. Several have a legitimate historical substitute,
    named in :data:`RATIONALE`.

Only ``POINT_IN_TIME_SAFE`` fields may be read directly from the row being scored.
:func:`assert_model_input_safe` enforces that at the boundary where a feature
matrix is handed to a model.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple

from simulator.leakage import (
    ENTITY_LABEL_FIELDS,
    LABEL_FIELDS,
    LeakageError,
    assert_feature_safe,
)
from simulator.transactions import TRANSACTION_FIELDS

__all__ = [
    "ENTITY_FIELD_ROLES",
    "FeatureContractError",
    "FieldRole",
    "RATIONALE",
    "TRANSACTION_FIELD_ROLES",
    "assert_model_input_safe",
    "fields_with_role",
    "role_of",
]


class FeatureContractError(LeakageError):
    """A field was read in a way the contract forbids.

    Subclasses :class:`simulator.leakage.LeakageError` so a single ``except``
    clause — or a single test — covers both the label barrier and this contract.
    """


class FieldRole(str, Enum):
    """How a field may be used. See the module docstring for each meaning."""

    LABEL = "label"
    IDENTIFIER = "identifier"
    POINT_IN_TIME_SAFE = "point_in_time_safe"
    AGGREGATE_ONLY = "aggregate_only"
    FORBIDDEN = "forbidden"


#: The only role that may be read directly from the row being scored.
MODEL_INPUT_ROLES: frozenset = frozenset({FieldRole.POINT_IN_TIME_SAFE})

_ID = FieldRole.IDENTIFIER
_PIT = FieldRole.POINT_IN_TIME_SAFE
_AGG = FieldRole.AGGREGATE_ONLY
_NO = FieldRole.FORBIDDEN
_LBL = FieldRole.LABEL

#: Every column of ``simulator.transactions.TRANSACTION_FIELDS``. Exhaustiveness is
#: asserted at import time, so a new column cannot be added to the schema without
#: being classified here.
TRANSACTION_FIELD_ROLES: Mapping[str, FieldRole] = {
    "transaction_id": _ID,
    "account_id": _ID,
    "customer_id": _ID,
    "merchant_id": _ID,
    "beneficiary_id": _ID,
    "device_id": _ID,
    "ip_id": _ID,
    "timestamp": _PIT,
    "amount": _PIT,
    "currency": _PIT,
    "channel": _PIT,
    "transaction_type": _PIT,
    "lat": _PIT,
    "lon": _PIT,
    "status": _AGG,
    "is_fraud": _LBL,
    "attack_id": _LBL,
}

#: Entity attributes, keyed by the entity name used in
#: :data:`simulator.leakage.ENTITY_LABEL_FIELDS`.
ENTITY_FIELD_ROLES: Mapping[str, Mapping[str, FieldRole]] = {
    "customer": {
        "customer_id": _ID,
        "created_at": _PIT,
        "country": _PIT,
        "region": _PIT,
        "home_lat": _PIT,
        "home_lon": _PIT,
        "segment": _PIT,
        "spending_profile": _NO,
        "risk_band": _PIT,
        "is_synthetic_identity": _LBL,
    },
    "merchant": {
        "merchant_id": _ID,
        "created_at": _PIT,
        "category": _PIT,
        "country": _PIT,
        "lat": _PIT,
        "lon": _PIT,
        "size_tier": _PIT,
        "avg_ticket_size": _NO,
        "is_fake": _LBL,
    },
    "account": {
        "account_id": _ID,
        "customer_id": _ID,
        "created_at": _PIT,
        "account_type": _PIT,
        "status": _NO,
        "baseline_tx_frequency": _NO,
        "baseline_amount_mean": _NO,
        "baseline_amount_std": _NO,
    },
    "device": {
        "device_id": _ID,
        "created_at": _PIT,
        "device_type": _PIT,
        "os_family": _PIT,
        "fingerprint_hash": _ID,
        "is_shared_legitimate": _LBL,
    },
    "ip": {
        "ip_id": _ID,
        "created_at": _PIT,
        "ip_address": _ID,
        "country": _PIT,
        "region": _PIT,
        "lat": _PIT,
        "lon": _PIT,
        "ip_type": _PIT,
        "is_shared_legitimate": _LBL,
    },
    "beneficiary": {
        "beneficiary_id": _ID,
        "created_at": _PIT,
        "beneficiary_type": _PIT,
        "country": _PIT,
        "linked_account_id": _ID,
    },
}

#: Why the non-obvious classifications are what they are. Kept short on purpose;
#: the reasoning matters more than the coverage, so only the fields where a
#: reviewer would reasonably disagree are listed.
RATIONALE: Mapping[str, str] = {
    "status": (
        "AGGREGATE_ONLY. A row's own settlement outcome (completed/pending/failed) "
        "is decided after the authorization being scored, so reading it off the row "
        "is reading the future. Earlier rows' statuses are observable and may be "
        "aggregated — e.g. an account's prior failure rate."
    ),
    "account.status": (
        "FORBIDDEN. Account state carries no change history in this dataset, so its "
        "value cannot be reconstructed as-of T. It happens to be static in the "
        "current generator, which is exactly what would make a feature depending on "
        "it silently leaky the day the field became mutable."
    ),
    "account.baseline_tx_frequency": (
        "FORBIDDEN. A generator parameter — the account's true activity rate, not an "
        "observable. Historical substitute: a velocity count over [T-Δ, T)."
    ),
    "account.baseline_amount_mean": (
        "FORBIDDEN. Generator parameter. Historical substitute: the account's mean "
        "amount over [T-Δ, T)."
    ),
    "account.baseline_amount_std": (
        "FORBIDDEN. Generator parameter. Historical substitute: the account's amount "
        "standard deviation over [T-Δ, T)."
    ),
    "merchant.avg_ticket_size": (
        "FORBIDDEN. Set at merchant creation from the size tier, and set explicitly "
        "for shell merchants — so it partly encodes the label. Historical "
        "substitute: the merchant's observed mean amount over [T-Δ, T)."
    ),
    "customer.spending_profile": (
        "FORBIDDEN. The generator's amount-law selector. Historical substitute: the "
        "customer's own amount distribution over [T-Δ, T)."
    ),
    "customer.risk_band": (
        "POINT_IN_TIME_SAFE. Derived at creation from spending volatility only, "
        "never from fraud status (simulator/entities.py). A real issuer holds an "
        "equivalent onboarding attribute."
    ),
    "merchant.size_tier": (
        "POINT_IN_TIME_SAFE but audit-watched. An onboarding attribute a real "
        "processor knows, and the Checkpoint 1 shortcut audit already probes it "
        "because it also fixes avg_ticket_size and merchant popularity."
    ),
    "timestamp": (
        "POINT_IN_TIME_SAFE. The as-of anchor itself. Calendar features derived from "
        "it (hour, day of week) are safe; anything comparing it to a later row is "
        "not."
    ),
    "device.fingerprint_hash": (
        "IDENTIFIER. A device pseudonym, usable for grouping history and never as a "
        "model input."
    ),
    "beneficiary.linked_account_id": (
        "IDENTIFIER. The edge that closes a circular-money-flow cycle. It is a join "
        "key for graph construction (Checkpoint 3), not a feature."
    ),
}


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------

#: Restrictiveness order, used only when a bare column name could refer to more
#: than one schema field. The guard resolves ambiguity pessimistically.
_SEVERITY: Tuple[FieldRole, ...] = (_LBL, _NO, _AGG, _ID, _PIT)


def _build_aliases() -> Dict[str, FieldRole]:
    """Map every plausible column spelling to its most restrictive role.

    A join produces ``merchant.avg_ticket_size`` or ``merchant_avg_ticket_size`` as
    readily as a bare ``avg_ticket_size``, and ``status`` alone is ambiguous between
    the transaction column and the account attribute. Collisions therefore resolve
    to whichever role is stricter, so the guard can only ever err toward refusing.
    """
    aliases: Dict[str, FieldRole] = {}

    def add(name: str, role: FieldRole) -> None:
        current = aliases.get(name)
        if current is None or _SEVERITY.index(role) < _SEVERITY.index(current):
            aliases[name] = role

    for name, role in TRANSACTION_FIELD_ROLES.items():
        add(name, role)
    for entity, fields in ENTITY_FIELD_ROLES.items():
        for name, role in fields.items():
            add(name, role)
            add(f"{entity}.{name}", role)
            add(f"{entity}_{name}", role)
    return aliases


_ALIASES: Mapping[str, FieldRole] = _build_aliases()


def role_of(field: str, *, entity: Optional[str] = None) -> Optional[FieldRole]:
    """The declared role of a schema field, or ``None`` if it is not one.

    Transaction columns win when ``entity`` is omitted, so ``role_of("status")``
    describes the transaction column and ``role_of("status", entity="account")``
    the account attribute. Unknown names return ``None`` rather than raising:
    derived feature names such as ``tx_velocity_1h`` are outside this contract by
    construction.
    """
    if entity is not None:
        return ENTITY_FIELD_ROLES.get(entity, {}).get(field)
    if field in TRANSACTION_FIELD_ROLES:
        return TRANSACTION_FIELD_ROLES[field]
    return _ALIASES.get(field)


def fields_with_role(
    role: FieldRole, *, entity: Optional[str] = None
) -> Tuple[str, ...]:
    """Every field carrying ``role``, on the transaction row or on one entity."""
    source: Mapping[str, FieldRole] = (
        TRANSACTION_FIELD_ROLES if entity is None else ENTITY_FIELD_ROLES.get(entity, {})
    )
    return tuple(name for name, r in source.items() if r is role)


def assert_model_input_safe(
    columns: Sequence[str], *, context: str = "feature matrix"
) -> None:
    """Fail unless every recognised schema column in ``columns`` is PIT-safe.

    Two checks, in order. First :func:`simulator.leakage.assert_feature_safe`, which
    owns the label barrier and its suffix matching. Then this contract, which
    additionally refuses identifiers, aggregate-only fields, and generator
    internals.

    Names the contract does not recognise pass through untouched. That is the point
    of the split: ``account_tx_velocity_1h`` is a *derived* feature whose safety is
    guaranteed by :mod:`ml.point_in_time`, not by a name check.
    """
    assert_feature_safe(columns, context=context)

    offenders: Dict[str, FieldRole] = {}
    for col in columns:
        role = _ALIASES.get(col) or _ALIASES.get(col.split(".")[-1])
        if role is not None and role not in MODEL_INPUT_ROLES:
            offenders[col] = role
    if offenders:
        detail = ", ".join(
            f"{col} ({role.value})" for col, role in sorted(offenders.items())
        )
        raise FeatureContractError(
            f"{context} reads field(s) the feature contract forbids as direct model "
            f"inputs: {detail}. See ml/feature_contract.py RATIONALE for the "
            f"historical substitute in each case."
        )


# --------------------------------------------------------------------------
# Import-time consistency
# --------------------------------------------------------------------------


def _check_exhaustive() -> None:
    """Fail fast if the contract has drifted from the schema it describes.

    Runs at import, so a schema change surfaces the moment anything imports this
    module rather than at the first feature that reads the unclassified column.
    """
    declared = set(TRANSACTION_FIELD_ROLES)
    actual = set(TRANSACTION_FIELDS)
    if declared != actual:
        raise FeatureContractError(
            "feature contract is out of sync with TRANSACTION_FIELDS: "
            f"unclassified={sorted(actual - declared)}, "
            f"unknown={sorted(declared - actual)}"
        )

    labelled = {n for n, r in TRANSACTION_FIELD_ROLES.items() if r is FieldRole.LABEL}
    if labelled != set(LABEL_FIELDS):
        raise FeatureContractError(
            "transaction LABEL roles disagree with simulator.leakage.LABEL_FIELDS: "
            f"{sorted(labelled)} != {sorted(LABEL_FIELDS)}"
        )

    for entity, fields in ENTITY_LABEL_FIELDS.items():
        here = {
            n for n, r in ENTITY_FIELD_ROLES.get(entity, {}).items()
            if r is FieldRole.LABEL
        }
        if here != set(fields):
            raise FeatureContractError(
                f"{entity} LABEL roles disagree with "
                f"simulator.leakage.ENTITY_LABEL_FIELDS: "
                f"{sorted(here)} != {sorted(fields)}"
            )


_check_exhaustive()
