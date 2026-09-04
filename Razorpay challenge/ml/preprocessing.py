"""Model-matrix assembly: turn the feature matrix into what a model may consume.

Two responsibilities, deliberately kept separate from feature engineering
(``ml/features.py``) and from any model:

1. Reassert, at this boundary too, that the matrix contains exactly the 25
   registered feature columns (``ml.feature_registry.FEATURE_COLUMNS``), that
   every value is finite, and that no label, attack-metadata, or identifier
   column is present (``ml.feature_contract.assert_model_input_safe``). This
   is defense in depth on top of the identical check
   ``ml.features.build_feature_matrix`` already performs, not a claim that
   the upstream guarantee is untrusted.

2. Provide the one preprocessing transform the project's own baseline ladder
   requires: standardization for Model 1, logistic regression
   (``docs/MODEL_BASELINES.md`` §3 -- "Scaling required"). Model 2, XGBoost,
   needs no scaling and is unaffected by this module. :class:`StandardScaler`
   is a concrete :class:`ml.split_policy.SplitSafeTransformer`, so "fit on
   train only, apply anywhere" is enforced by the base class, not by
   convention here: ``fit`` refuses every split but TRAIN and refuses a
   second call; ``transform`` refuses to run before ``fit``.

No imputation is implemented here. ``docs/MODEL_BASELINES.md`` §3 states
cold-start handling lives in the feature layer with documented defaults,
"not by a generic imputer" -- and ``build_feature_matrix`` already guarantees
no feature ever emits NaN or infinity. A non-finite value reaching this
module is therefore treated as a bug upstream and raises loudly rather than
being silently patched. No encoding is implemented either: every column in
the registered 25-feature contract is already numeric.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ml.feature_contract import assert_model_input_safe
from ml.feature_registry import FEATURE_COLUMNS
from ml.split_policy import SplitSafeTransformer

__all__ = ["MIN_SCALE_STD", "StandardScaler", "assert_model_matrix_safe"]

#: Below this, a TRAIN column is treated as numerically constant: centered but
#: not scaled, matching the standard convention (e.g. scikit-learn's
#: StandardScaler) for avoiding division by an amount indistinguishable from
#: zero, rather than manufacturing an arbitrarily large scaled value out of
#: floating-point noise.
MIN_SCALE_STD: float = 1e-8


def assert_model_matrix_safe(matrix: pd.DataFrame, *, context: str = "model matrix") -> None:
    """Fail unless ``matrix`` is exactly the registered 25-feature contract and finite.

    Checks, in order: no label/attack-metadata/identifier/aggregate-only/
    forbidden column is present (``assert_model_input_safe``, which raises
    ``simulator.leakage.LeakageError`` or its subclass
    ``ml.feature_contract.FeatureContractError``); the column set and order
    match ``FEATURE_COLUMNS`` exactly; every value is finite.
    """
    assert_model_input_safe(matrix.columns.tolist(), context=context)
    if list(matrix.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            f"{context} does not match the registered feature contract. "
            f"expected={list(FEATURE_COLUMNS)}, got={list(matrix.columns)}"
        )
    if not matrix.empty and not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError(f"{context} contains NaN or infinity")


class StandardScaler(SplitSafeTransformer):
    """Zero-mean, unit-variance scaling of the 25-feature matrix, fit on TRAIN only.

    Required for Model 1 (logistic regression) per ``docs/MODEL_BASELINES.md``
    §3; Model 2 (XGBoost) does not use it. Deterministic: fitting and
    transforming are pure column-wise arithmetic over the input DataFrame,
    with no randomness and no dependency on row order.

    A column with (numerically) zero variance in TRAIN is centered but left
    unscaled -- divided by ``1.0``, not by a near-zero standard deviation --
    so a constant training column cannot blow up into an arbitrarily large
    value from floating-point noise.
    """

    def __init__(self) -> None:
        super().__init__(name="StandardScaler")
        self._mean: pd.Series | None = None
        self._scale: pd.Series | None = None
        self._columns: tuple[str, ...] | None = None

    @property
    def mean_(self) -> Mapping[str, float]:
        if self._mean is None:
            raise RuntimeError(f"{self.name} is not fitted")
        return dict(self._mean)

    @property
    def scale_(self) -> Mapping[str, float]:
        if self._scale is None:
            raise RuntimeError(f"{self.name} is not fitted")
        return dict(self._scale)

    def _fit(self, data: pd.DataFrame) -> None:
        assert_model_matrix_safe(data, context=f"{self.name}.fit input")
        mean = data.mean(axis=0)
        # Population standard deviation (ddof=0): this is a scaling parameter
        # derived from the whole TRAIN split, not an inferential estimate of a
        # larger population -- the same convention scikit-learn's
        # StandardScaler uses, and what makes TRAIN's own distribution map to
        # unit variance exactly.
        std = data.std(axis=0, ddof=0)
        scale = std.where(std > MIN_SCALE_STD, other=1.0)
        self._mean = mean
        self._scale = scale
        self._columns = tuple(data.columns)

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert_model_matrix_safe(data, context=f"{self.name}.transform input")
        if tuple(data.columns) != self._columns:
            raise ValueError(
                f"{self.name}.transform received columns {list(data.columns)}, "
                f"fitted on {list(self._columns)}"
            )
        transformed = (data - self._mean) / self._scale
        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise ValueError(f"{self.name}.transform produced NaN or infinity")
        return transformed
