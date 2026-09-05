"""Checkpoint 2E: train and evaluate the Model 0 / Model 1 baseline ladder.

Loads warm-up, train, and validation only -- heldout is never named or
imported (see PERMITTED_SPLITS below and docs/DATA_SPLIT_PROTOCOL.md).
Builds the 25-feature matrix exactly as scripts/benchmark_feature_generation.py
does, fits preprocessing and both models on TRAIN only, evaluates and selects
an operating threshold on VALIDATION only under the primary cost
configuration, and persists the fitted preprocessing, both models, and one
experiment record.

No feature is added, removed, or redefined. No resampling of any kind. No
graph features. No held-out access. No documentation is updated by this
script -- measured numbers belong in docs/MODEL_BASELINES.md and
docs/COST_MODEL.md only after a real run's output has been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # ships with scikit-learn; no new dependency added
import numpy as np

from ml.evaluation import evaluate_on_validation
from ml.evaluation_contract import cost_config_from_env
from ml.feature_registry import FEATURE_COLUMNS
from ml.features import build_feature_matrix
from ml.models import LogisticRegressionBaseline, MajorityClassifier
from ml.preprocessing import StandardScaler, assert_model_matrix_safe
from simulator.transactions import Transaction

DATA_ROOT = REPO_ROOT / "data"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
#: Exactly the splits this script may ever read. "heldout" is never here.
PERMITTED_SPLITS = ("warmup", "train", "validation")


def _transaction(row: dict[str, str]) -> Transaction:
    return Transaction(
        transaction_id=row["transaction_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        account_id=row["account_id"],
        customer_id=row["customer_id"],
        merchant_id=row["merchant_id"] or None,
        beneficiary_id=row["beneficiary_id"] or None,
        device_id=row["device_id"],
        ip_id=row["ip_id"],
        amount=float(row["amount"]),
        currency=row["currency"],
        channel=row["channel"],
        transaction_type=row["transaction_type"],
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        status=row["status"],
        # Labels are read here, in the training script, as the supervised
        # target -- never inside ml/features.py, ml/temporal.py, or
        # ml/point_in_time.py, none of which are touched by this checkpoint.
        is_fraud=row["is_fraud"].strip().lower() in {"true", "1"},
        attack_id=row["attack_id"] or None,
    )


def _read_transactions(split: str) -> list[Transaction]:
    if split not in PERMITTED_SPLITS:
        raise ValueError(f"unpermitted split: {split}")
    path = DATA_ROOT / "splits" / split / "transactions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [_transaction(row) for row in csv.DictReader(handle)]


def _read_accounts() -> dict[str, dict[str, datetime]]:
    path = DATA_ROOT / "raw" / "accounts.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["account_id"]: {"created_at": datetime.fromisoformat(row["created_at"])}
            for row in csv.DictReader(handle)
        }


def _labels(transactions: list[Transaction]) -> np.ndarray:
    return np.array([t.is_fraud for t in transactions], dtype=bool)


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _generate_experiment_id(config: dict, now: datetime) -> str:
    """Filesystem-safe, unique-per-invocation experiment identifier.

    Combines a microsecond-resolution rendering of ``now`` with
    ``_config_hash(config)`` (unchanged), so two runs sharing an identical
    config -- the scenario that previously caused a silent artifact
    collision -- still receive different ids, since they cannot share the
    same microsecond. ``_config_hash`` alone is deliberately kept unchanged
    and is still what the record's own ``config_hash`` field reports.
    """
    run_slug = now.strftime("%Y%m%dT%H%M%S%f")
    return f"{run_slug}_{_config_hash(config)}"


def _assert_artifact_does_not_exist(path: Path) -> None:
    """Refuse to silently overwrite an existing artifact.

    A collision here would mean ``experiment_id`` was reused rather than
    freshly generated. This is the fail-loudly complement to
    ``_generate_experiment_id``'s timestamp-based uniqueness, not a
    substitute for it.
    """
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing artifact at {path}; "
            "experiment_id must not collide with an existing experiment"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SENTINEL_SEED", 42)))
    parser.add_argument("--git-commit", type=str, default="unknown")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_ROOT)
    args = parser.parse_args()

    warmup = _read_transactions("warmup")
    train = _read_transactions("train")
    validation = _read_transactions("validation")
    accounts = _read_accounts()

    print("building train features", flush=True)
    train_matrix = build_feature_matrix(
        train, history_transactions=[*warmup, *train], accounts=accounts
    )
    print("building validation features", flush=True)
    validation_matrix = build_feature_matrix(
        validation, history_transactions=[*warmup, *train, *validation], accounts=accounts
    )
    assert_model_matrix_safe(train_matrix, context="train feature matrix")
    assert_model_matrix_safe(validation_matrix, context="validation feature matrix")

    y_train = _labels(train)
    y_validation = _labels(validation)

    scaler = StandardScaler().fit(train_matrix, split="train")
    train_scaled = scaler.transform(train_matrix, split="train")
    validation_scaled = scaler.transform(validation_matrix, split="validation")

    majority = MajorityClassifier().fit(train_matrix, y_train, split="train")
    logreg = LogisticRegressionBaseline(random_state=args.seed).fit(
        train_scaled, y_train, split="train"
    )

    majority_scores = majority.predict_proba(validation_matrix)
    logreg_scores = logreg.predict_proba(validation_scaled)

    cost = cost_config_from_env()
    majority_report = evaluate_on_validation(
        y_validation, majority_scores, model_type="majority", cost=cost
    )
    logreg_report = evaluate_on_validation(
        y_validation, logreg_scores, model_type="logistic_regression", cost=cost
    )

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    config = {
        "seed": args.seed,
        "feature_columns": list(FEATURE_COLUMNS),
        "cost_config": cost.name,
    }
    experiment_id = _generate_experiment_id(config, now)

    preprocessing_dir = args.output_dir / "preprocessing"
    models_dir = args.output_dir / "models"
    reports_dir = args.output_dir / "reports"
    for directory in (preprocessing_dir, models_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    scaler_path = preprocessing_dir / f"standard_scaler_{experiment_id}.json"
    _assert_artifact_does_not_exist(scaler_path)
    scaler_path.write_text(
        json.dumps(
            {
                "mean": scaler.mean_,
                "scale": scaler.scale_,
                "columns": list(FEATURE_COLUMNS),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    majority_path = models_dir / f"majority_classifier_{experiment_id}.joblib"
    logreg_path = models_dir / f"logistic_regression_{experiment_id}.joblib"
    _assert_artifact_does_not_exist(majority_path)
    _assert_artifact_does_not_exist(logreg_path)
    joblib.dump(majority, majority_path)
    joblib.dump(logreg, logreg_path)

    # Hyperparameters are read from each model's own live configuration, not
    # duplicated as a separate hardcoded dict, so this cannot drift from what
    # was actually fit. LogisticRegressionBaseline already exposes its
    # configuration publicly via `.hyperparameters` (ml/models.py, unchanged).
    # MajorityClassifier has no public hyperparameters accessor -- ml/models.py
    # is out of scope for this fix -- so its underlying
    # sklearn.dummy.DummyClassifier's own get_params() is read directly; this
    # is scikit-learn's own introspection of the actual fitted estimator, not
    # a restated copy.
    majority_record = {
        **majority_report.to_dict(),
        "hyperparameters": majority._model.get_params(),
    }
    logreg_record = {**logreg_report.to_dict(), "hyperparameters": logreg.hyperparameters}

    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "git_commit": args.git_commit,
        "seed": args.seed,
        "config_hash": _config_hash(config),
        "dataset_version": (
            "local data/ directory -- see docs/DATA_SPLIT_PROTOCOL.md \u00a78 for the "
            "reference generation this project documents (seed 42, 5% prevalence)"
        ),
        "split": "validation",
        "feature_set": "B - transaction + temporal (25 registered features; no graph/reputation yet)",
        "preprocessing_ref": str(scaler_path.relative_to(REPO_ROOT)),
        "models": [majority_record, logreg_record],
        "notes": "Checkpoint 2E baseline ladder (Model 0 + Model 1). Heldout not accessed.",
    }
    report_path = reports_dir / f"{experiment_id}.json"
    _assert_artifact_does_not_exist(report_path)
    report_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"experiment_id": experiment_id, "report_path": str(report_path)}))


if __name__ == "__main__":
    main()
