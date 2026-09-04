"""Abuse-Ring Sentinel — synthetic ecosystem and fraud simulator.

This package generates the entire synthetic dataset: entities, normal
transaction behaviour, seven parameterized attack families, and the
ground-truth registry.

**All data produced here is synthetic.** No real person, account, device, or
financial record is represented anywhere in this package.

Import note: :mod:`simulator.ground_truth` is deliberately *not* re-exported
here. Ground truth is evaluation-only and must never be reachable from feature,
model, or scoring code. See ``docs/FEATURE_LEAKAGE_POLICY.md`` §4.
"""

from simulator.config import (
    AttackMix,
    EcosystemConfig,
    SimulationConfig,
    SplitConfig,
    default_config,
)

__all__ = [
    "AttackMix",
    "EcosystemConfig",
    "SimulationConfig",
    "SplitConfig",
    "default_config",
]
