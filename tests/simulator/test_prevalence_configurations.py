"""7. Fraud prevalence configuration — all four benchmark rates execute and land.

The four supported prevalences are a *benchmark sweep*, not claims about real-world
fraud rates: 5% is the primary synthetic benchmark configuration, and 1% / 10% / 25%
exist so that later checkpoints can show how detection behaves as the positive class
gets rarer or denser. The regression risk is that a generator change quietly breaks one
end of the sweep — the 1% budget starving a family, or the 25% budget exhausting the
entity pool — while the primary configuration keeps passing.

Tolerance is read from ``config.prevalence_tolerance`` and applied exactly the way
:func:`simulator.validation.validate_dataset` applies it: a *relative* deviation from
target, measured over the evaluation window only. Warm-up is excluded because it holds
no fraud by design, so including it would drag the realised rate down mechanically.
Restating the rule with a hand-picked absolute tolerance is what this file is written to
avoid.

Realised prevalence is read from the validator's own measurement rather than recomputed,
so the test cannot disagree with the report the project publishes.

These tests are the expensive ones in the suite — four generations — so each prevalence
is generated once and shared with the shortcut-audit regression through the session
cache in ``conftest.py``.
"""

from __future__ import annotations

import pytest

from simulator.config import SPLIT_NAMES, SUPPORTED_PREVALENCES, default_config
from simulator.ground_truth import ATTACK_FAMILIES
from simulator.validation import validate_dataset


def test_the_supported_sweep_is_the_expected_four_rates():
    """A freeze guard: the benchmark sweep itself must not drift."""
    assert SUPPORTED_PREVALENCES == (0.01, 0.05, 0.10, 0.25)


def test_unsupported_prevalence_is_rejected_at_configuration_time():
    """Fail loudly rather than silently generate an unbenchmarked dataset."""
    with pytest.raises(ValueError):
        default_config(seed=42, prevalence=0.5)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("prevalence", SUPPORTED_PREVALENCES)
def test_configuration_generates_and_validates(make_dataset, prevalence):
    ds = make_dataset(prevalence=prevalence)
    report = validate_dataset(ds)

    assert ds.transactions, "generation produced no transactions"
    assert report.ok, (
        f"prevalence {prevalence:.0%} failed validation: "
        f"{[i.code for i in report.errors]}"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("prevalence", SUPPORTED_PREVALENCES)
def test_realised_prevalence_is_within_the_documented_tolerance(make_dataset, prevalence):
    ds = make_dataset(prevalence=prevalence)
    report = validate_dataset(ds)

    target = report.measurements["prevalence_target"]
    realised = report.measurements["prevalence_realised_evaluation_window"]
    tol = ds.config.prevalence_tolerance

    assert target == pytest.approx(prevalence)
    assert abs(realised - target) / target <= tol, (
        f"realised {realised:.4f} deviates from target {target:.4f} by more than "
        f"the documented {tol:.0%} relative tolerance"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("prevalence", SUPPORTED_PREVALENCES)
def test_every_family_appears_in_every_split_at_every_prevalence(make_dataset, prevalence):
    """The seven-family floor is what makes the 1% end of the sweep tight.

    ``docs/ML_READINESS.md`` §5 requires every split to contain at least one instance
    of all seven families, so one instance of each is a *floor* on a split's fraud row
    count and therefore a floor on achievable prevalence. A starved budget would drop a
    family, and since Class A and Class B are reported separately throughout the
    project, that removes a reporting category rather than merely shrinking the data.

    ``validate_dataset`` flags this too; asserted here so the failure names the split
    and the family instead of an error code.
    """
    ds = make_dataset(prevalence=prevalence)

    missing = {
        split: sorted(set(ATTACK_FAMILIES) - {r.attack_type for r in ds.ground_truth.by_split(split)})
        for split in SPLIT_NAMES
    }
    missing = {split: families for split, families in missing.items() if families}

    assert not missing, f"prevalence {prevalence:.0%} is missing families: {missing}"
