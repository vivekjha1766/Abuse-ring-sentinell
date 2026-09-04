"""Which split each operation may touch, and the mechanism that enforces it.

``docs/DATA_SPLIT_PROTOCOL.md`` §3 states the permissions matrix in prose and §4
states how the held-out set is sealed. This module is that text in executable form,
so the rules fail a call rather than a code review.

Three failure modes are the ones that actually happen, and each has a guard:

*Fitting on validation or held-out.* :data:`PERMISSIONS` permits
``FIT_MODEL``, ``FIT_PREPROCESSING`` and ``CLASS_RATIO`` on ``TRAIN`` alone, and
:meth:`SplitSafeTransformer.fit` refuses anything else.

*Selecting a threshold on held-out.* ``THRESHOLD_SELECTION`` is permitted on
``VALIDATION`` alone. :mod:`ml.evaluation_contract` routes every threshold decision
through that check.

*Fitting preprocessing globally.* :func:`assert_permitted` accepts a *collection*
of observed splits, so a caller that hands over the splits actually present in its
data gets a failure the moment more than ``TRAIN`` appears. A transformer also
refuses a second :meth:`~SplitSafeTransformer.fit`, which is how a global refit
usually arrives — as an innocuous-looking second call.

The warm-up split is permitted for nothing except supplying history. It is not
scored and is not part of the 60/20/20 split (``docs/DATA_SPLIT_PROTOCOL.md`` §5).
"""

from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar, Union

from simulator.config import SPLIT_NAMES
from simulator.pipeline import WARMUP_SPLIT

__all__ = [
    "FIT_SPLIT",
    "Operation",
    "PERMISSIONS",
    "SealedSplitError",
    "Split",
    "SplitPolicyError",
    "SplitSafeTransformer",
    "TUNING_SPLIT",
    "assert_permitted",
    "is_permitted",
    "unseal_heldout",
]

T = TypeVar("T")


class SplitPolicyError(AssertionError):
    """An operation touched a split it is not permitted to touch.

    An ``AssertionError`` subclass for the same reason as
    :class:`simulator.leakage.LeakageError`: this is a broken invariant, and code
    that catches it and continues has invalidated the evaluation.
    """


class SealedSplitError(SplitPolicyError):
    """The held-out split was reached without being explicitly unsealed."""


class Split(str, Enum):
    """The four partitions. Values match the strings the simulator emits."""

    WARMUP = "warmup"
    TRAIN = "train"
    VALIDATION = "validation"
    HELDOUT = "heldout"


class Operation(str, Enum):
    """Things one can do with data. The rows of ``DATA_SPLIT_PROTOCOL`` §3."""

    FIT_MODEL = "fit_model"
    FIT_PREPROCESSING = "fit_preprocessing"
    CLASS_RATIO = "class_ratio"
    HYPERPARAMETER_SEARCH = "hyperparameter_search"
    EARLY_STOPPING = "early_stopping"
    THRESHOLD_SELECTION = "threshold_selection"
    MODEL_SELECTION = "model_selection"
    FINAL_REPORTING = "final_reporting"
    #: Applying already-fitted parameters. Not in §3 because it consumes nothing
    #: from the split it is applied to — the parameters came from TRAIN.
    TRANSFORM = "transform"
    #: Supplying earlier rows as feature history. Permitted everywhere because it
    #: is governed by point-in-time safety (ts < as_of), not by split membership;
    #: see DATA_SPLIT_PROTOCOL §5 on entity spillover.
    PROVIDE_HISTORY = "provide_history"


_ALL: frozenset = frozenset(Split)

#: The permissions matrix. Held-out appears exactly once, for final reporting.
PERMISSIONS: Mapping[Operation, frozenset] = {
    Operation.FIT_MODEL: frozenset({Split.TRAIN}),
    Operation.FIT_PREPROCESSING: frozenset({Split.TRAIN}),
    Operation.CLASS_RATIO: frozenset({Split.TRAIN}),
    Operation.HYPERPARAMETER_SEARCH: frozenset({Split.VALIDATION}),
    Operation.EARLY_STOPPING: frozenset({Split.VALIDATION}),
    Operation.THRESHOLD_SELECTION: frozenset({Split.VALIDATION}),
    Operation.MODEL_SELECTION: frozenset({Split.VALIDATION}),
    Operation.FINAL_REPORTING: frozenset({Split.HELDOUT}),
    Operation.TRANSFORM: _ALL,
    Operation.PROVIDE_HISTORY: _ALL,
}

#: The only split anything may be fitted on.
FIT_SPLIT: Split = Split.TRAIN

#: The only split anything may be tuned or thresholded on.
TUNING_SPLIT: Split = Split.VALIDATION

#: Opened once, at Checkpoint 6, through :func:`unseal_heldout`.
SEALED_SPLITS: frozenset = frozenset({Split.HELDOUT})

#: Not scored, not part of the 60/20/20 split; exists only to give the longest
#: feature window real history on the first evaluation day.
HISTORY_ONLY_SPLITS: frozenset = frozenset({Split.WARMUP})

SplitLike = Union[Split, str]


def as_split(value: SplitLike) -> Split:
    """Coerce a split name to :class:`Split`, failing loudly on a typo."""
    if isinstance(value, Split):
        return value
    try:
        return Split(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise SplitPolicyError(
            f"unknown split {value!r}; expected one of {[s.value for s in Split]}"
        ) from exc


def is_permitted(operation: Operation, split: SplitLike) -> bool:
    """Whether ``operation`` may read ``split``."""
    return as_split(split) in PERMISSIONS[operation]


def assert_permitted(
    operation: Operation,
    splits: Union[SplitLike, Iterable[SplitLike]],
    *,
    context: str = "",
) -> Split:
    """Fail unless ``operation`` may read every split in ``splits``.

    Accepts one split or a collection of them. The collection form is what catches
    global preprocessing: pass the splits actually present in the data and fitting
    fails as soon as the data is not train-only. Returns the single split when
    exactly one was supplied and permitted, so callers can record what they used.
    """
    if isinstance(splits, (Split, str)):
        observed = [as_split(splits)]
    else:
        observed = [as_split(s) for s in splits]
    if not observed:
        raise SplitPolicyError(
            f"{operation.value} was asked to run on no split at all"
            + (f" ({context})" if context else "")
        )

    allowed = PERMISSIONS[operation]
    offending = sorted({s.value for s in observed if s not in allowed})
    if offending:
        where = f" ({context})" if context else ""
        raise SplitPolicyError(
            f"{operation.value} is not permitted on {offending}{where}. "
            f"Permitted: {sorted(s.value for s in allowed)}. "
            "See docs/DATA_SPLIT_PROTOCOL.md §3."
        )
    if len(set(observed)) > 1 and allowed is not _ALL:
        raise SplitPolicyError(
            f"{operation.value} spans multiple splits "
            f"{sorted({s.value for s in observed})}{f' ({context})' if context else ''}; "
            "it must run on exactly one."
        )
    return observed[0]


# --------------------------------------------------------------------------
# Split-safe preprocessing
# --------------------------------------------------------------------------


class SplitSafeTransformer(ABC):
    """Base class for any scaler, encoder, or imputer.

    Subclasses implement :meth:`_fit` and :meth:`_transform` and inherit the policy:
    fit on train only, fit once, transform anywhere. Validation and held-out are
    therefore transform-only by construction rather than by discipline.

    No concrete transformer lives here — the actual scaling and imputation is
    Checkpoint 2 work (``docs/FEATURE_REFERENCE.md``). This is the contract those
    classes will inherit.
    """

    def __init__(self, *, name: Optional[str] = None):
        self._name = name or type(self).__name__
        self._fitted_on: Optional[Split] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def fitted_on(self) -> Optional[Split]:
        """The split this transformer was fitted on, or ``None`` if unfitted."""
        return self._fitted_on

    @property
    def is_fitted(self) -> bool:
        return self._fitted_on is not None

    def fit(
        self, data: Any, *, split: Union[SplitLike, Iterable[SplitLike]]
    ) -> "SplitSafeTransformer":
        """Fit on train data only, once.

        Refitting is refused rather than allowed-with-a-warning. A second fit is how
        preprocessing quietly becomes global: the first call is on train, the second
        on "all the data" for convenience. If parameters genuinely need to change,
        construct a new transformer so the change is visible in the code.
        """
        if self.is_fitted:
            raise SplitPolicyError(
                f"{self._name} is already fitted on {self._fitted_on.value!r}; "
                "refitting would make the preprocessing global. Construct a new "
                "transformer instead."
            )
        resolved = assert_permitted(
            Operation.FIT_PREPROCESSING, split, context=f"{self._name}.fit"
        )
        self._fit(data)
        self._fitted_on = resolved
        return self

    def transform(self, data: Any, *, split: SplitLike) -> Any:
        """Apply train-fitted parameters to any split."""
        if not self.is_fitted:
            raise SplitPolicyError(
                f"{self._name}.transform called before fit; there are no parameters "
                "to apply."
            )
        assert_permitted(Operation.TRANSFORM, split, context=f"{self._name}.transform")
        return self._transform(data)

    def fit_transform(self, data: Any, *, split: SplitLike) -> Any:
        """Convenience for the train split only; the guard is inherited from fit."""
        self.fit(data, split=split)
        return self.transform(data, split=split)

    @abstractmethod
    def _fit(self, data: Any) -> None:
        """Compute and store parameters from train data."""

    @abstractmethod
    def _transform(self, data: Any) -> Any:
        """Apply stored parameters. Must not update them."""


# --------------------------------------------------------------------------
# Sealing the held-out split
# --------------------------------------------------------------------------


def unseal_heldout(
    loader: Callable[[], T], *, reason: str, unseal: bool = False
) -> T:
    """Load held-out data, refusing by default (``DATA_SPLIT_PROTOCOL`` §4).

    The default refuses, an explicit ``unseal=True`` is required, a non-empty
    ``reason`` is required, and a warning naming the calling file and line is
    emitted on every successful unseal. The warning is the point: a held-out read
    that happens before Checkpoint 6 leaves a trace in the test and CI output
    instead of passing silently.

    ``loader`` is a zero-argument callable so this guard can wrap whatever the
    Checkpoint 6 loader turns out to be without importing it here.
    """
    if not unseal:
        raise SealedSplitError(
            "the held-out split is sealed until Checkpoint 6. It may not be used for "
            "fitting, tuning, threshold selection, or sanity checks. Pass "
            "unseal=True with a reason to open it for final reporting "
            "(docs/DATA_SPLIT_PROTOCOL.md §4)."
        )
    if not reason or not reason.strip():
        raise SealedSplitError("unsealing the held-out split requires a stated reason")

    frame = inspect.stack()[1]
    warnings.warn(
        f"HELD-OUT SPLIT UNSEALED by {frame.filename}:{frame.lineno} — {reason.strip()}",
        UserWarning,
        stacklevel=2,
    )
    return loader()


# --------------------------------------------------------------------------
# Import-time consistency
# --------------------------------------------------------------------------


def _check_split_names() -> None:
    """Fail fast if these split names have drifted from the simulator's."""
    expected = {WARMUP_SPLIT, *SPLIT_NAMES}
    declared = {s.value for s in Split}
    if declared != expected:
        raise SplitPolicyError(
            "ml.split_policy.Split has drifted from the simulator's split names: "
            f"{sorted(declared)} != {sorted(expected)}"
        )
    missing = [op.value for op in Operation if op not in PERMISSIONS]
    if missing:
        raise SplitPolicyError(f"operations with no permissions entry: {missing}")


_check_split_names()
