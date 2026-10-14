"""Split isolation: fit on train, tune on validation, never touch held-out.

``docs/DATA_SPLIT_PROTOCOL.md`` §3 is a table, and a table in a document does not
stop anyone. These tests check that the executable form actually refuses the three
things the protocol forbids — fitting on validation or held-out, selecting a
threshold on held-out, and fitting preprocessing globally — plus the two mechanical
ways a global fit sneaks in: a second ``fit`` call, and a ``fit`` handed data that
spans splits.

The permissions matrix itself is asserted literally. That test exists to make a
future relaxation deliberate: loosening the protocol should require editing a test
that spells out the old rule, not just editing the rule.
"""

from __future__ import annotations

import warnings

import pytest

from ml.split_policy import (
    FIT_SPLIT,
    HISTORY_ONLY_SPLITS,
    PERMISSIONS,
    SEALED_SPLITS,
    TUNING_SPLIT,
    Operation,
    SealedSplitError,
    Split,
    SplitPolicyError,
    SplitSafeTransformer,
    assert_permitted,
    unseal_heldout,
)

pytestmark = pytest.mark.leakage

#: ``docs/DATA_SPLIT_PROTOCOL.md`` §3, transcribed. Restated here on purpose.
EXPECTED_PERMISSIONS = {
    Operation.FIT_MODEL: {"train"},
    Operation.FIT_PREPROCESSING: {"train"},
    Operation.CLASS_RATIO: {"train"},
    Operation.HYPERPARAMETER_SEARCH: {"validation"},
    Operation.EARLY_STOPPING: {"validation"},
    Operation.THRESHOLD_SELECTION: {"validation"},
    Operation.MODEL_SELECTION: {"validation"},
    Operation.FINAL_REPORTING: {"heldout"},
    Operation.TRANSFORM: {"warmup", "train", "validation", "heldout"},
    Operation.PROVIDE_HISTORY: {"warmup", "train", "validation", "heldout"},
}


class RecordingScaler(SplitSafeTransformer):
    """Minimal concrete transformer: records what it was fitted on.

    Stands in for the Checkpoint 2 scalers and imputers, which do not exist yet. The
    policy under test lives in the base class, so a subclass this small exercises it
    fully.
    """

    def __init__(self):
        super().__init__(name="RecordingScaler")
        self.fit_calls: list = []
        self.mean: float = 0.0

    def _fit(self, data) -> None:
        self.fit_calls.append(list(data))
        self.mean = sum(data) / len(data)

    def _transform(self, data):
        return [x - self.mean for x in data]


def test_permissions_matrix_matches_the_protocol():
    """A relaxation must be spelled out here before it can take effect."""
    actual = {op: {s.value for s in splits} for op, splits in PERMISSIONS.items()}
    assert actual == EXPECTED_PERMISSIONS
    assert FIT_SPLIT is Split.TRAIN
    assert TUNING_SPLIT is Split.VALIDATION
    assert SEALED_SPLITS == frozenset({Split.HELDOUT})
    assert HISTORY_ONLY_SPLITS == frozenset({Split.WARMUP})


@pytest.mark.parametrize("split", ["validation", "heldout", "warmup"])
def test_preprocessing_cannot_be_fitted_outside_train(split):
    """The failure mode: a scaler fitted on everything "for stability"."""
    scaler = RecordingScaler()
    with pytest.raises(SplitPolicyError):
        scaler.fit([1.0, 2.0, 3.0], split=split)
    assert scaler.fit_calls == []
    assert not scaler.is_fitted


@pytest.mark.parametrize("split", ["train", "validation", "heldout", "warmup"])
def test_transform_is_permitted_on_every_split(split):
    """Applying train-fitted parameters consumes nothing from the target split."""
    scaler = RecordingScaler().fit([0.0, 10.0], split="train")
    assert scaler.transform([10.0], split=split) == [5.0]
    assert scaler.fitted_on is Split.TRAIN


def test_fitting_across_splits_is_refused():
    """Global preprocessing, caught by declaring the splits the data spans."""
    scaler = RecordingScaler()
    with pytest.raises(SplitPolicyError):
        scaler.fit([1.0, 2.0], split=["train", "validation"])
    with pytest.raises(SplitPolicyError):
        assert_permitted(
            Operation.FIT_PREPROCESSING,
            ["train", "validation", "heldout"],
            context="global scaler",
        )
    assert not scaler.is_fitted


def test_refitting_is_refused():
    """The second call is how a train-only fit quietly becomes a global one."""
    scaler = RecordingScaler().fit([1.0, 2.0], split="train")
    with pytest.raises(SplitPolicyError):
        scaler.fit([1.0, 2.0, 100.0], split="train")
    assert len(scaler.fit_calls) == 1
    assert scaler.mean == 1.5


def test_transform_before_fit_is_refused():
    """There are no parameters to apply, and silently applying none is worse."""
    with pytest.raises(SplitPolicyError):
        RecordingScaler().transform([1.0], split="validation")


def test_threshold_selection_is_validation_only():
    """The second protocol violation, at the policy layer."""
    assert_permitted(Operation.THRESHOLD_SELECTION, "validation")
    for split in ("train", "heldout", "warmup"):
        with pytest.raises(SplitPolicyError):
            assert_permitted(Operation.THRESHOLD_SELECTION, split)


def test_heldout_is_sealed_by_default():
    """No accidental read: the loader is not even called."""
    calls = []

    def loader():
        calls.append(1)
        return "heldout rows"

    with pytest.raises(SealedSplitError):
        unseal_heldout(loader, reason="just a quick look")
    assert calls == []

    for op in (Operation.FIT_MODEL, Operation.FIT_PREPROCESSING, Operation.THRESHOLD_SELECTION):
        with pytest.raises(SplitPolicyError):
            assert_permitted(op, "heldout")


def test_unsealing_requires_a_reason_and_leaves_a_trace():
    """When it is finally opened at Checkpoint 6, the read is loud and attributed."""
    with pytest.raises(SealedSplitError):
        unseal_heldout(lambda: None, reason="   ", unseal=True)

    # warnings.catch_warnings rather than pytest.warns: this suite must run under a
    # minimal runner as well as under real pytest.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = unseal_heldout(
            lambda: "heldout rows",
            reason="final Checkpoint 6 evaluation",
            unseal=True,
        )

    assert result == "heldout rows"
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "HELD-OUT SPLIT UNSEALED" in message
    assert "final Checkpoint 6 evaluation" in message
    assert "test_split_policy.py" in message


def test_warmup_is_history_only():
    """Not scored, not fitted on, not tuned on — only a source of prior rows."""
    assert_permitted(Operation.PROVIDE_HISTORY, "warmup")
    for op in (
        Operation.FIT_MODEL,
        Operation.FIT_PREPROCESSING,
        Operation.CLASS_RATIO,
        Operation.HYPERPARAMETER_SEARCH,
        Operation.THRESHOLD_SELECTION,
        Operation.FINAL_REPORTING,
    ):
        with pytest.raises(SplitPolicyError):
            assert_permitted(op, "warmup")


def test_an_unknown_split_name_fails_loudly():
    """A typo must not silently become an unguarded operation."""
    with pytest.raises(SplitPolicyError):
        assert_permitted(Operation.FIT_MODEL, "trian")
    with pytest.raises(SplitPolicyError):
        assert_permitted(Operation.FIT_MODEL, [])
