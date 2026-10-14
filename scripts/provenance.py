"""Deterministic provenance helpers for new training experiments.

This module is deliberately independent of model fitting and held-out loading.
Callers choose the files to hash; the training script supplies warm-up, train,
validation, and account metadata only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PROVENANCE_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "xgboost",
    "networkx",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible data with stable ordering and separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def hash_json(value: Any) -> str:
    """Return the full SHA-256 digest of a JSON-compatible value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def hash_file(path: Path) -> str:
    """Hash one file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    """Hash an explicit set of dataset inputs in deterministic path order."""
    files = []
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        files.append({"path": relative, "sha256": hash_file(path), "size": path.stat().st_size})
    return {"files": files, "content_hash": hash_json(files)}


def split_metadata(split_transactions: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    """Capture observed non-heldout split ranges and row counts."""
    metadata: dict[str, Any] = {}
    for split, transactions in split_transactions.items():
        timestamps = [transaction.timestamp for transaction in transactions]
        metadata[split] = {
            "rows": len(transactions),
            "min_timestamp": min(timestamps).isoformat() if timestamps else None,
            "max_timestamp": max(timestamps).isoformat() if timestamps else None,
        }
    return metadata


def package_versions(package_names: Sequence[str] = PROVENANCE_PACKAGES) -> dict[str, str]:
    """Return installed versions, recording unavailable optional packages explicitly."""
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_commit(repo_root: Path) -> str:
    """Return the current commit, or an explicit unavailable marker."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed git command and trusted repo root
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def runtime_metadata() -> dict[str, Any]:
    """Capture interpreter, platform, dependency, and deterministic settings."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "environment": {
            key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
        },
        "sys_executable": sys.executable,
    }


def build_provenance(
    *,
    experiment_id: str,
    repo_root: Path,
    dataset_files: Iterable[Path],
    dataset_manifest: Mapping[str, Any] | None,
    resolved_config: Mapping[str, Any],
    feature_columns: Sequence[str],
    split_transactions: Mapping[str, Sequence[Any]],
    seed: int,
    threshold_method: str,
    cost: Mapping[str, Any],
    runtime_settings: Mapping[str, Any] | None = None,
    supplied_git_commit: str = "",
) -> dict[str, Any]:
    """Assemble the provenance manifest for one new experiment."""
    manifest_identity = None
    if dataset_manifest is not None:
        manifest_identity = {
            "config_fingerprint": dataset_manifest.get("config_fingerprint"),
            "content_hash": dataset_manifest.get("content_hash"),
            "boundaries": dataset_manifest.get("boundaries"),
        }
    return {
        "experiment_id": experiment_id,
        "git_commit": supplied_git_commit or git_commit(repo_root),
        "dataset": {
            "identity": dataset_identity(dataset_files, root=repo_root),
            "manifest": manifest_identity,
        },
        "resolved_config": dict(resolved_config),
        "feature_schema": {"columns": list(feature_columns), "count": len(feature_columns)},
        "splits": split_metadata(split_transactions),
        "seed": seed,
        "threshold_selection": {
            "method": threshold_method,
            "split": "validation",
        },
        "cost": dict(cost),
        "runtime": runtime_metadata(),
        "runtime_settings": dict(runtime_settings or {}),
        "created_at": datetime.now().astimezone().isoformat(),
    }