"""Machine-learning layer: point-in-time features, models, evaluation.

Checkpoint 1.5 populates only the *safety scaffolding* of this package — the
rules a feature must obey and the mechanism that makes obeying them the path of
least resistance:

* :mod:`ml.feature_contract` — which fields may be read, and how.
* :mod:`ml.point_in_time` — the only sanctioned way to look at history.
* :mod:`ml.split_policy` — which split each operation may touch.
* :mod:`ml.evaluation_contract` — cost model, metric primacy, threshold policy.

No features, no models, and no training code exist here yet; those arrive at
Checkpoint 2 (``docs/FEATURE_REFERENCE.md``, ``docs/MODEL_BASELINES.md``).

Nothing in this package may import :mod:`simulator.ground_truth`, and no module
here that computes features may read ``is_fraud`` or ``attack_id``. The barrier is
one-way by design and asserted by ``tests/leakage/test_feature_contract.py``
rather than left to convention.

Submodules are deliberately not re-exported here. Importing ``ml`` must stay
cheap and dependency-free so that a leakage test never drags in a model library.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
