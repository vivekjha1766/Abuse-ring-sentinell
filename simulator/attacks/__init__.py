"""Attack family registry.

Maps each of the seven family names to its ``generate`` function. The pipeline
looks families up by the same string keys used everywhere else — ``ATTACK_FAMILIES``
in :mod:`simulator.ground_truth`, the ``AttackMix`` fields, and the
``SplitConfig.params_for`` accessor — so there is exactly one spelling of each
family name across the whole simulator.

Keeping this indirection here (rather than a big ``if/elif`` in the pipeline)
means a family is fully described by its own module plus one registry line, and
the pipeline stays agnostic to how any particular attack is built.
"""

from __future__ import annotations

from typing import Dict

from simulator.attacks.base import (
    AttackContext,
    AttackResult,
    AttackGenerator,
)
from simulator.attacks import (
    account_takeover,
    circular_money_flow,
    fake_merchant_ring,
    microtransaction_attack,
    shared_device_ring,
    synthetic_identity_cluster,
    transaction_spike,
)

__all__ = [
    "ATTACK_GENERATORS",
    "AttackContext",
    "AttackResult",
    "generate_attack",
]

#: family name -> its generator function.
ATTACK_GENERATORS: Dict[str, AttackGenerator] = {
    "fake_merchant_ring": fake_merchant_ring.generate,
    "shared_device_ring": shared_device_ring.generate,
    "transaction_spike": transaction_spike.generate,
    "circular_money_flow": circular_money_flow.generate,
    "synthetic_identity_cluster": synthetic_identity_cluster.generate,
    "microtransaction_attack": microtransaction_attack.generate,
    "account_takeover": account_takeover.generate,
}


def generate_attack(family: str, ctx: AttackContext, params: object) -> AttackResult:
    """Dispatch to the named family's generator.

    Raising on an unknown family (rather than silently skipping) keeps a
    mis-spelled family name from quietly reducing the injected fraud count.
    """
    try:
        generator = ATTACK_GENERATORS[family]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown attack family: {family!r}") from exc
    return generator(ctx, params)
