"""The feature contract refuses what it says it refuses.

A field-role table is only worth having if something enforces it, so these tests
exercise the enforcement rather than the table. Three properties matter:

*The contract cannot silently fall behind the schema.* Exhaustiveness against
``TRANSACTION_FIELDS`` and agreement with ``simulator.leakage`` are asserted at
import; these tests assert the same thing from outside, so the guarantee is visible
in the suite rather than only in a module that happens to be imported.

*The guard refuses each unsafe category for its own reason.* Labels, identifiers,
aggregate-only fields, and generator internals are four different mistakes; each is
tested separately.

*Derived feature names still pass.* A contract that refused
``account_tx_velocity_1h`` would be abandoned at the first real feature, so the
pass-through case is tested as deliberately as the refusals.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ml.feature_contract import (
    ENTITY_FIELD_ROLES,
    RATIONALE,
    TRANSACTION_FIELD_ROLES,
    FeatureContractError,
    FieldRole,
    assert_model_input_safe,
    fields_with_role,
    role_of,
)
from simulator.leakage import ENTITY_LABEL_FIELDS, LABEL_FIELDS, LeakageError
from simulator.transactions import TRANSACTION_FIELDS

pytestmark = pytest.mark.leakage

#: Repository root, three levels up from ``tests/leakage/test_feature_contract.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_classifies_every_transaction_column():
    """A new column cannot enter the schema unclassified."""
    assert set(TRANSACTION_FIELD_ROLES) == set(TRANSACTION_FIELDS)


def test_label_roles_agree_with_the_simulator():
    """Ground truth is defined once, in ``simulator.leakage``, and mirrored here."""
    assert set(fields_with_role(FieldRole.LABEL)) == set(LABEL_FIELDS)
    for entity, fields in ENTITY_LABEL_FIELDS.items():
        assert set(fields_with_role(FieldRole.LABEL, entity=entity)) == set(fields)


def test_point_in_time_safe_columns_are_accepted():
    """The safe set is usable — the guard is not vacuously strict."""
    safe = fields_with_role(FieldRole.POINT_IN_TIME_SAFE)
    assert "amount" in safe and "channel" in safe
    assert_model_input_safe(list(safe), context="pit-safe columns")


def test_labels_are_refused_by_the_reused_barrier():
    """``simulator.leakage`` owns this refusal; the contract inherits its exception."""
    with pytest.raises(LeakageError):
        assert_model_input_safe(["amount", "is_fraud"])
    with pytest.raises(LeakageError):
        assert_model_input_safe(["amount", "attack_id"])


@pytest.mark.parametrize(
    "identifier",
    ["transaction_id", "account_id", "device_id", "ip_id", "fingerprint_hash"],
)
def test_identifiers_are_refused_as_model_inputs(identifier):
    """IDs group history; as inputs they encode generator ordering, not behaviour."""
    with pytest.raises(FeatureContractError):
        assert_model_input_safe(["amount", identifier])


def test_the_rows_own_status_is_refused_but_history_is_not():
    """Settlement outcome is decided after the authorization being scored."""
    with pytest.raises(FeatureContractError):
        assert_model_input_safe(["amount", "status"])

    assert role_of("status") is FieldRole.AGGREGATE_ONLY
    assert role_of("status", entity="account") is FieldRole.FORBIDDEN
    # An aggregate over earlier rows' statuses is a derived name and passes.
    assert_model_input_safe(["amount", "account_prior_failure_rate_7d"])


@pytest.mark.parametrize(
    "column",
    [
        "baseline_amount_mean",
        "baseline_tx_frequency",
        "account.baseline_amount_std",
        "account_status",
        "avg_ticket_size",
        "merchant.avg_ticket_size",
        "spending_profile",
    ],
)
def test_generator_internals_are_refused_however_they_are_spelled(column):
    """A join renames things; the guard matches bare, dotted, and prefixed forms."""
    with pytest.raises(FeatureContractError):
        assert_model_input_safe(["amount", column])


def test_ambiguous_bare_names_resolve_to_the_stricter_role():
    """``status`` belongs to two schemas; the guard must take the harsher reading."""
    with pytest.raises(FeatureContractError):
        assert_model_input_safe(["status"])
    with pytest.raises(FeatureContractError):
        assert_model_input_safe(["account.status"])


def test_derived_feature_names_pass_through():
    """Point-in-time safety, not name matching, guarantees these."""
    assert_model_input_safe(
        [
            "amount",
            "tx_velocity_15m",
            "amount_deviation",
            "geo_distance_from_prev",
            "account_age_days",
            "device_account_degree",
        ],
        context="planned checkpoint 2 features",
    )
    assert role_of("tx_velocity_15m") is None


def test_no_ml_module_reads_ground_truth():
    """The label barrier as a structural property of the package.

    An AST scan rather than a text search, so the module docstrings that *discuss*
    ``is_fraud`` and the string dict keys that *classify* it do not register as
    reads. What is forbidden is an import of the ground-truth module and an
    attribute access such as ``row.is_fraud``.
    """
    offences = []
    for path in sorted((REPO_ROOT / "ml").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("simulator.ground_truth"):
                        offences.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("simulator.ground_truth"):
                    offences.append(f"{path.name}: from {node.module} import ...")
            elif isinstance(node, ast.Attribute):
                if node.attr in LABEL_FIELDS:
                    offences.append(f"{path.name}:{node.lineno}: .{node.attr}")
    assert offences == []


def test_every_forbidden_field_has_a_recorded_rationale():
    """A refusal without a stated reason and a substitute gets overridden later."""
    undocumented = []
    for entity, fields in ENTITY_FIELD_ROLES.items():
        for name, role in fields.items():
            if role is FieldRole.FORBIDDEN and f"{entity}.{name}" not in RATIONALE:
                undocumented.append(f"{entity}.{name}")
    for name, role in TRANSACTION_FIELD_ROLES.items():
        if role in (FieldRole.FORBIDDEN, FieldRole.AGGREGATE_ONLY) and name not in RATIONALE:
            undocumented.append(name)
    assert undocumented == []
