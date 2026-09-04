"""The two implemented rungs of the baseline ladder (docs/MODEL_BASELINES.md).

Model 0 (majority/prior) and Model 1 (logistic regression) are wrapped in a
thin split-policy-aware layer: ``.fit(X, y, split=...)`` calls
``ml.split_policy.assert_permitted(Operation.FIT_MODEL, split)`` before
touching scikit-learn at all, so fitting on anything but TRAIN raises
``SplitPolicyError`` before a single parameter is estimated -- the same
mechanism ``ml.preprocessing.StandardScaler`` inherits from
``SplitSafeTransformer``, applied here to models instead of transformers.
This module only *calls* ``ml.split_policy``'s already-public
``assert_permitted``; ``ml/split_policy.py`` itself is not modified.

Model 2 (XGBoost) is out of scope for this checkpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from ml.preprocessing import assert_model_matrix_safe
from ml.split_policy import Operation, Split, SplitLike, SplitPolicyError, assert_permitted

__all__ = ["LogisticRegressionBaseline", "MajorityClassifier"]


class _SplitSafeModel:
    """Shared fit-on-train-only, refuse-refit, refuse-predict-before-fit policy.

    A small local reuse of ``ml.split_policy.assert_permitted`` for models,
    not an addition to ``ml/split_policy.py`` (which owns the transformer
    contract, ``SplitSafeTransformer``, and is off limits for this
    checkpoint).
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._fitted_on: Split | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._fitted_on is not None

    @property
    def fitted_on(self) -> Split | None:
        return self._fitted_on

    def fit(self, X: pd.DataFrame, y: Any, *, split: SplitLike) -> "_SplitSafeModel":
        if self.is_fitted:
            raise SplitPolicyError(
                f"{self._name} is already fitted on {self._fitted_on.value!r}; "
                "refitting would make training global. Construct a new model "
                "instead."
            )
        assert_model_matrix_safe(X, context=f"{self._name}.fit input")
        resolved = assert_permitted(Operation.FIT_MODEL, split, context=f"{self._name}.fit")
        self._fit(X, y)
        self._fitted_on = resolved
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise SplitPolicyError(
                f"{self._name}.predict_proba called before fit; there is no fitted "
                "model to score with."
            )
        assert_model_matrix_safe(X, context=f"{self._name}.predict_proba input")
        return self._predict_proba(X)

    def _fit(self, X: pd.DataFrame, y: Any) -> None:
        raise NotImplementedError

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class MajorityClassifier(_SplitSafeModel):
    """Model 0: the majority/prior baseline (docs/MODEL_BASELINES.md).

    Wraps ``sklearn.dummy.DummyClassifier(strategy="prior")``. Per spec,
    "Inputs: None (predicts the majority class)" and "Preprocessing: None" --
    ``X`` is accepted only for scikit-learn API compatibility (row count /
    shape) and its *values* never influence the prediction:
    ``strategy="prior"`` always returns the class prior computed from ``y``
    at fit time, for every row, regardless of what ``X`` contains. This is
    asserted directly in ``tests/leakage/test_models.py`` by calling
    ``predict_proba`` with two different-valued ``X`` matrices of the same
    shape and checking the outputs are identical.
    """

    def __init__(self) -> None:
        super().__init__(name="MajorityClassifier")
        self._model = DummyClassifier(strategy="prior")

    def _fit(self, X: pd.DataFrame, y: Any) -> None:
        self._model.fit(X, y)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class LogisticRegressionBaseline(_SplitSafeModel):
    """Model 1: logistic regression (docs/MODEL_BASELINES.md).

    Expects an already-scaled matrix -- "Scaling required" per spec, fit via
    ``ml.preprocessing.StandardScaler`` on TRAIN beforehand; this class does
    not scale internally, keeping the preprocessing and modelling layers
    separate.

    Class imbalance is addressed via ``class_weight="balanced"``
    (docs/MODEL_BASELINES.md: "Class imbalance addressed via class
    weighting") -- scikit-learn's standard, documented mechanism, computed
    from whatever ``y`` is passed to ``fit`` (TRAIN only, enforced by the
    base class). No resampling of any kind (no SMOTE, no other technique) is
    used anywhere in this module.
    """

    def __init__(
        self,
        *,
        random_state: int,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        penalty: str = "l2",
    ) -> None:
        super().__init__(name="LogisticRegressionBaseline")
        self._hyperparameters: dict[str, Any] = {
            "class_weight": "balanced",
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "penalty": penalty,
            "random_state": random_state,
        }
        self._model = LogisticRegression(**self._hyperparameters)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return dict(self._hyperparameters)

    @property
    def coefficients(self) -> dict[str, float]:
        if not self.is_fitted:
            raise SplitPolicyError(f"{self.name} is not fitted")
        names = list(self._model.feature_names_in_)
        values = list(self._model.coef_[0])
        return dict(zip(names, values, strict=True))

    def _fit(self, X: pd.DataFrame, y: Any) -> None:
        self._model.fit(X, y)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]
