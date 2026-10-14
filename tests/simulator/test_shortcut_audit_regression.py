"""8. Shortcut-audit regression — zero oracle findings at all four prevalences.

The shortcut audit is the reason this dataset can be trusted at all. It sweeps every
candidate leak — amount, velocity, account age, geography, entity creation time, sharing
structure, merchant profile, generator metadata, attack-specific fields — against every
attack family and reports any feature that separates fraud from normal almost perfectly.
Anything at or above ``ORACLE_AUC`` fails the audit outright: it means the generator, not
the modelling, would be producing the metrics.

**What this file asserts, and what it very deliberately does not.**

It asserts the audit finds *zero oracles* at 1%, 5%, 10% and 25% — the frozen Checkpoint
1 result. It does **not** assert that strong signals must disappear. Strong signals are
expected and wanted: a dataset in which no feature carries real signal would be
undetectable by construction and useless for the modelling checkpoints. The audit's own
verdict scale encodes exactly this — ``strong`` passes, ``oracle`` fails.

In particular, ``device_account_degree`` → ``account_takeover`` is retained as an
intentional non-oracle strong signal (AUC 0.9554 / 0.9010 / 0.8006 / 0.6286 across the
sweep, all below the 0.98 gate). The test below pins that it is still *measured* and
still *not an oracle*. Asserting it must be weak would be asserting the takeover family
should stop behaving like a takeover.

The gate constants are pinned too. Without that, a future change could make this suite
pass by moving the goalposts rather than by fixing the data — the one failure mode a
regression test for an audit cannot detect from its own results.
"""

from __future__ import annotations

import pytest

from simulator.config import SUPPORTED_PREVALENCES
from simulator.shortcut_audit import (
    ORACLE_AUC,
    ORACLE_PRECISION,
    ORACLE_RECALL,
    STRONG_AUC,
)

#: The signal Checkpoint 1 measured, inspected and consciously kept.
RETAINED_STRONG_SIGNAL = ("device_account_degree", "account_takeover")


def test_audit_gates_are_frozen():
    """A freeze guard: the suite must not be satisfiable by relaxing the gates."""
    assert ORACLE_AUC == 0.98
    assert STRONG_AUC == 0.90
    assert ORACLE_PRECISION == 0.95
    assert ORACLE_RECALL == 0.50


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("prevalence", SUPPORTED_PREVALENCES)
def test_no_oracle_features_at_any_benchmark_prevalence(audit_for, prevalence):
    report = audit_for(prevalence)

    offenders = [
        f"{r.feature} -> {r.target} (auc={r.auc}, f1={r.best_f1:.3f}, rule={r.rule})"
        for r in report.oracles
    ]

    assert not offenders, f"oracle shortcuts at {prevalence:.0%}: {offenders}"
    assert report.ok, report.summary()
    # Also from the unfiltered result list, in case `oracles` ever stops filtering.
    assert not [r for r in report.results if r.verdict == "oracle"]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("prevalence", SUPPORTED_PREVALENCES)
def test_audit_actually_ran_its_probes(audit_for, prevalence):
    """An audit that silently stopped probing would report zero oracles too.

    The count is asserted as a floor, not an exact value: a pair is skipped when its
    usable rows or positives fall below the probe's minimum, so the total legitimately
    moves by a few tests between prevalences.
    """
    report = audit_for(prevalence)

    assert len(report.results) >= 400, (
        f"only {len(report.results)} shortcut tests ran at {prevalence:.0%}"
    )
    assert report.measurements["rows_audited"] > 0
    assert all(count > 0 for count in report.measurements["targets"].values())
    assert {r.verdict for r in report.results} <= {
        "oracle",
        "strong",
        "ok",
        "quarantined",  # label fields, probed on purpose — see test_label_quarantine
    }


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("prevalence", SUPPORTED_PREVALENCES)
def test_retained_strong_signal_is_measured_and_below_the_oracle_gate(
    audit_for, prevalence
):
    """``device_account_degree`` → ``account_takeover`` stays measured, stays legal.

    Two failure modes are guarded, in opposite directions: the probe quietly
    disappearing (so nobody would notice if it drifted upward), and the signal crossing
    the oracle gate. Its magnitude below that gate is *not* constrained.
    """
    feature, target = RETAINED_STRONG_SIGNAL
    report = audit_for(prevalence)

    matches = [r for r in report.results if r.feature == feature and r.target == target]

    assert matches, f"{feature} -> {target} was not probed at {prevalence:.0%}"
    for result in matches:
        assert result.auc is not None
        assert result.auc < ORACLE_AUC, (
            f"{feature} -> {target} became an oracle at {prevalence:.0%} "
            f"(auc={result.auc:.4f})"
        )
        assert not result.is_oracle
