"""Checkpoint 2H: final, one-time HELDOUT evaluation of the frozen 2G models.

This script fits nothing. It loads the exact preprocessing and model
artifacts persisted by one specific, already-completed
``scripts/train_baseline_models.py`` run (identified by ``--experiment-id``),
applies each model's already-selected VALIDATION threshold -- read verbatim
from that run's own experiment report, never recomputed -- and evaluates on
HELDOUT. Heldout is opened exactly once here, through
``ml.split_policy.unseal_heldout()``, the single sanctioned gate
(``docs/DATA_SPLIT_PROTOCOL.md`` §4), and is used for nothing except
building point-in-time-safe features for itself (as history, exactly like
validation used ``[warmup, train, validation]``) and computing the final
metrics below. Nothing in this file selects a threshold, fits a model, or
fits preprocessing against heldout -- there is no call to
``select_threshold`` here at all, and ``ml.split_policy`` would refuse one
against heldout regardless.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # ships with scikit-learn; no new dependency added
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evaluation_contract import (
    PRIMARY_COST,
    PRIMARY_RANKING_METRIC,
    ConfusionMatrix,
    assert_metric_report_valid,
)
from ml.features import build_feature_matrix
from ml.preprocessing import assert_model_matrix_safe
from ml.split_policy import unseal_heldout
from scripts.train_baseline_models import _assert_artifact_does_not_exist
from simulator.transactions import Transaction

DATA_ROOT = REPO_ROOT / "data"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
#: The ordinary reader below may only ever be pointed at these. heldout is
#: read exactly once, elsewhere, only through unseal_heldout().
PERMITTED_ORDINARY_SPLITS = ("warmup", "train", "validation")


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
        is_fraud=row["is_fraud"].strip().lower() in {"true", "1"},
        attack_id=row["attack_id"] or None,
    )


def _read_split_csv(path: Path) -> list[Transaction]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [_transaction(row) for row in csv.DictReader(handle)]


def _read_transactions(split: str) -> list[Transaction]:
    if split not in PERMITTED_ORDINARY_SPLITS:
        raise ValueError(f"unpermitted split for the ordinary reader: {split}")
    return _read_split_csv(DATA_ROOT / "splits" / split / "transactions.csv")


def _read_heldout_transactions() -> list[Transaction]:
    """The one sanctioned read of heldout in this codebase's evaluation path."""
    return unseal_heldout(
        lambda: _read_split_csv(DATA_ROOT / "splits" / "heldout" / "transactions.csv"),
        reason="Checkpoint 2H: final, one-time heldout evaluation of the frozen 2G models",
        unseal=True,
    )


def _read_accounts() -> dict[str, dict[str, datetime]]:
    path = DATA_ROOT / "raw" / "accounts.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["account_id"]: {"created_at": datetime.fromisoformat(row["created_at"])}
            for row in csv.DictReader(handle)
        }


def _labels(transactions: list[Transaction]) -> np.ndarray:
    return np.array([t.is_fraud for t in transactions], dtype=bool)


def _load_source_report(experiment_id: str) -> dict:
    path = ARTIFACTS_ROOT / "reports" / f"{experiment_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no experiment report found at {path}; pass the exact --experiment-id "
            "of the frozen Checkpoint 2G run to evaluate on heldout"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_frozen_scaling(matrix: pd.DataFrame, scaler_params: dict) -> pd.DataFrame:
    """Replicate ml.preprocessing.StandardScaler._transform from persisted params.

    ``(data - mean) / scale``, using the exact mean/scale this run's scaler
    was fitted with on TRAIN -- read from disk, not refit here.
    """
    mean = pd.Series(scaler_params["mean"])
    scale = pd.Series(scaler_params["scale"])
    return (matrix - mean) / scale


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _evaluate_on_heldout(
    *, model_type: str, y_true: np.ndarray, scores: np.ndarray, frozen_threshold: float
) -> dict:
    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    confusion = ConfusionMatrix.from_predictions(y_true, scores >= frozen_threshold)
    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": confusion.precision,
        "recall": confusion.recall,
        "f1": confusion.f1,
        "confusion_matrix": confusion.to_dict(),
        "total_error_cost": confusion.total_error_cost(PRIMARY_COST),
    }
    assert_metric_report_valid(
        list(metrics), primary=PRIMARY_RANKING_METRIC, context=f"{model_type} heldout report"
    )
    return {
        "model_type": model_type,
        "metrics": metrics,
        "applied_validation_threshold": frozen_threshold,
        "cost_config": PRIMARY_COST.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-id",
        required=True,
        help=(
            "the frozen Checkpoint 2G experiment_id whose preprocessing/model "
            "artifacts and validation thresholds are evaluated on heldout"
        ),
    )
    parser.add_argument("--git-commit", type=str, default="unknown")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_ROOT)
    args = parser.parse_args()

    source_report = _load_source_report(args.experiment_id)
    threshold_by_model = {
        entry["model_type"]: entry["threshold"]["threshold"] for entry in source_report["models"]
    }
    hyperparameters_by_model = {
        entry["model_type"]: entry["hyperparameters"] for entry in source_report["models"]
    }

    warmup = _read_transactions("warmup")
    train = _read_transactions("train")
    validation = _read_transactions("validation")
    accounts = _read_accounts()
    heldout = _read_heldout_transactions()  # the one sanctioned heldout read

    print("building heldout features", flush=True)
    heldout_matrix = build_feature_matrix(
        heldout,
        history_transactions=[*warmup, *train, *validation, *heldout],
        accounts=accounts,
    )
    assert_model_matrix_safe(heldout_matrix, context="heldout feature matrix")
    y_heldout = _labels(heldout)

    preprocessing_dir = ARTIFACTS_ROOT / "preprocessing"
    models_dir = ARTIFACTS_ROOT / "models"
    scaler_path = preprocessing_dir / f"standard_scaler_{args.experiment_id}.json"
    scaler_params = json.loads(scaler_path.read_text(encoding="utf-8"))
    heldout_scaled = _apply_frozen_scaling(heldout_matrix, scaler_params)
    assert_model_matrix_safe(heldout_scaled, context="scaled heldout feature matrix")

    majority = joblib.load(models_dir / f"majority_classifier_{args.experiment_id}.joblib")
    logreg = joblib.load(models_dir / f"logistic_regression_{args.experiment_id}.joblib")
    xgboost_model = joblib.load(models_dir / f"xgboost_{args.experiment_id}.joblib")

    majority_scores = majority.predict_proba(heldout_matrix)
    logreg_scores = logreg.predict_proba(heldout_scaled)
    xgboost_scores = xgboost_model.predict_proba(heldout_matrix)

    majority_record = {
        **_evaluate_on_heldout(
            model_type="majority",
            y_true=y_heldout,
            scores=majority_scores,
            frozen_threshold=threshold_by_model["majority"],
        ),
        "hyperparameters": hyperparameters_by_model["majority"],
    }
    logreg_record = {
        **_evaluate_on_heldout(
            model_type="logistic_regression",
            y_true=y_heldout,
            scores=logreg_scores,
            frozen_threshold=threshold_by_model["logistic_regression"],
        ),
        "hyperparameters": hyperparameters_by_model["logistic_regression"],
    }
    xgboost_record = {
        **_evaluate_on_heldout(
            model_type="xgboost",
            y_true=y_heldout,
            scores=xgboost_scores,
            frozen_threshold=threshold_by_model["xgboost"],
        ),
        "hyperparameters": hyperparameters_by_model["xgboost"],
    }

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    report_id = f"heldout_final_{now.strftime('%Y%m%dT%H%M%S%f')}_{args.experiment_id}"

    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment_id": report_id,
        "source_experiment_id": args.experiment_id,
        "timestamp": timestamp,
        "git_commit": args.git_commit,
        "config_hash": _config_hash(
            {"source_experiment_id": args.experiment_id, "purpose": "checkpoint_2h_heldout"}
        ),
        "dataset_version": source_report.get("dataset_version"),
        "split": "heldout",
        "feature_set": source_report.get("feature_set"),
        "preprocessing_ref": source_report.get("preprocessing_ref"),
        "model_refs": {
            "majority": str(
                (models_dir / f"majority_classifier_{args.experiment_id}.joblib").relative_to(
                    REPO_ROOT
                )
            ),
            "logistic_regression": str(
                (models_dir / f"logistic_regression_{args.experiment_id}.joblib").relative_to(
                    REPO_ROOT
                )
            ),
            "xgboost": str(
                (models_dir / f"xgboost_{args.experiment_id}.joblib").relative_to(REPO_ROOT)
            ),
        },
        "models": [majority_record, logreg_record, xgboost_record],
        "heldout_access": {
            "purpose": "final_evaluation_only",
            "unsealed_via": "ml.split_policy.unseal_heldout",
            "fitting_performed_against_heldout": False,
            "threshold_selection_performed_against_heldout": False,
        },
        "notes": (
            "Checkpoint 2H final heldout evaluation. All three models and the "
            "preprocessing scaler are the exact frozen artifacts from "
            f"experiment_id={args.experiment_id}; nothing was refit. Each "
            "model's threshold was selected on VALIDATION during that prior "
            "run and is applied here unchanged. Heldout was opened exactly "
            "once, through unseal_heldout(), for this final evaluation only."
        ),
    }

    report_path = reports_dir / f"{report_id}.json"
    _assert_artifact_does_not_exist(report_path)
    report_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"experiment_id": report_id, "report_path": str(report_path)}))


if __name__ == "__main__":
    main()
