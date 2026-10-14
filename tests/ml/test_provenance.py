"""Focused tests for new-experiment provenance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.provenance import (
    build_provenance,
    dataset_identity,
    hash_json,
    split_metadata,
)


@dataclass(frozen=True)
class _Transaction:
    timestamp: object


def test_hash_json_is_order_independent():
    assert hash_json({"b": 2, "a": [1, 2]}) == hash_json({"a": [1, 2], "b": 2})


def test_dataset_identity_is_deterministic_and_tracks_file_changes(tmp_path: Path):
    first = tmp_path / "train.csv"
    second = tmp_path / "validation.csv"
    first.write_text("a\n1\n", encoding="utf-8")
    second.write_text("a\n2\n", encoding="utf-8")

    identity = dataset_identity([second, first], root=tmp_path)
    assert [item["path"] for item in identity["files"]] == ["train.csv", "validation.csv"]
    original_hash = identity["content_hash"]

    second.write_text("a\n3\n", encoding="utf-8")
    assert dataset_identity([first, second], root=tmp_path)["content_hash"] != original_hash


def test_split_metadata_records_counts_and_observed_ranges():
    from datetime import UTC, datetime

    first = _Transaction(datetime(2026, 1, 1, tzinfo=UTC))
    second = _Transaction(datetime(2026, 1, 2, tzinfo=UTC))
    metadata = split_metadata({"train": [second, first], "validation": []})

    assert metadata["train"]["rows"] == 2
    assert metadata["train"]["min_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert metadata["train"]["max_timestamp"] == "2026-01-02T00:00:00+00:00"
    assert metadata["validation"]["min_timestamp"] is None


def test_build_provenance_contains_required_future_run_fields(tmp_path: Path):
    dataset_file = tmp_path / "train.csv"
    dataset_file.write_text("transaction_id\ntx_1\n", encoding="utf-8")
    provenance = build_provenance(
        experiment_id="exp_new",
        repo_root=tmp_path,
        dataset_files=[dataset_file],
        dataset_manifest={
            "config_fingerprint": "cfg123",
            "content_hash": "data123",
            "boundaries": {"train_end": "2026-01-02T00:00:00+00:00"},
        },
        resolved_config={"seed": 42, "prevalence": 0.05},
        feature_columns=["feature_a"],
        split_transactions={"train": []},
        seed=42,
        threshold_method="validation_cost_minimization",
        cost={"name": "primary", "fp_cost": 15.0, "fn_cost": 450.0},
        supplied_git_commit="commit123",
    )

    assert provenance["experiment_id"] == "exp_new"
    assert provenance["git_commit"] == "commit123"
    assert provenance["dataset"]["manifest"]["config_fingerprint"] == "cfg123"
    assert provenance["feature_schema"] == {"columns": ["feature_a"], "count": 1}
    assert provenance["threshold_selection"]["split"] == "validation"
    assert provenance["cost"]["fn_cost"] == 450.0
    assert "python" in provenance["runtime"]