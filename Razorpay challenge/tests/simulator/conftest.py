"""Shared fixtures for the Checkpoint 1 simulator regression suite.

Generating a dataset is the expensive part of every test here — between roughly 13
and 45 seconds depending on prevalence — so datasets are built at most once per
session and memoised by ``(prevalence, seed)``. Tests receive the *same* object,
which is safe because nothing in this suite mutates a dataset: these are read-only
invariant checks over a frozen generator.

The suite deliberately covers only the nine areas listed in ``ROADMAP.md`` for the
Checkpoint 1 freeze. It asserts invariants through the existing public APIs
(:mod:`simulator.pipeline`, :mod:`simulator.validation`, :mod:`simulator.leakage`,
:mod:`simulator.shortcut_audit`) rather than re-deriving generator behaviour, so a
test can only fail if a real invariant broke.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import pytest

from simulator.config import SimulationConfig, default_config
from simulator.pipeline import GeneratedDataset, generate_dataset
from simulator.shortcut_audit import ShortcutAuditReport, audit_shortcuts

#: The primary synthetic benchmark configuration (``PRD.md`` §8) — a benchmark
#: choice, not a real-world or industry fraud rate.
PRIMARY_PREVALENCE = 0.05

#: The seed every frozen-checkpoint measurement was taken at.
SEED = 42


@pytest.fixture(scope="session")
def config_for() -> Callable[..., SimulationConfig]:
    """Build a configuration the same way every measurement in the project did."""

    def _config(
        prevalence: float = PRIMARY_PREVALENCE, seed: int = SEED
    ) -> SimulationConfig:
        return default_config(seed=seed, prevalence=prevalence)

    return _config


@pytest.fixture(scope="session")
def make_dataset(
    config_for: Callable[..., SimulationConfig],
) -> Callable[..., GeneratedDataset]:
    """Memoised dataset factory, keyed by ``(prevalence, seed)``.

    Returning a cached instance is what keeps the suite bounded: the prevalence
    sweep and the shortcut-audit regression would otherwise generate the same four
    datasets twice over.
    """
    cache: Dict[Tuple[float, int], GeneratedDataset] = {}

    def _make(
        prevalence: float = PRIMARY_PREVALENCE, seed: int = SEED
    ) -> GeneratedDataset:
        key = (prevalence, seed)
        if key not in cache:
            cache[key] = generate_dataset(config_for(prevalence, seed))
        return cache[key]

    return _make


@pytest.fixture(scope="session")
def dataset(make_dataset: Callable[..., GeneratedDataset]) -> GeneratedDataset:
    """The primary 5% dataset at seed 42 — the subject of most integrity checks."""
    return make_dataset()


@pytest.fixture(scope="session")
def audit_for(make_dataset: Callable[..., GeneratedDataset]) -> Callable[..., Any]:
    """Memoised shortcut-audit reports, keyed by prevalence.

    The audit is the most expensive operation in the suite — several hundred probe
    fits per dataset — and both the audit-regression file and the quarantine file
    interrogate the same reports, so it runs once per prevalence rather than once per
    test.
    """
    cache: Dict[float, ShortcutAuditReport] = {}

    def _audit(prevalence: float = PRIMARY_PREVALENCE) -> ShortcutAuditReport:
        if prevalence not in cache:
            cache[prevalence] = audit_shortcuts(make_dataset(prevalence=prevalence))
        return cache[prevalence]

    return _audit
