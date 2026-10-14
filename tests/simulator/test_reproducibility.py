"""1. Deterministic reproducibility — same seed and configuration, same dataset.

``content_hash`` covers the transaction rows only (labels included, deliberately),
so reproducibility of the *ecosystem* and the *ground-truth registry* is asserted
separately here. A generator can be row-identical and still have shuffled its entity
tables or renumbered its attack ids, and any of those would break the manifest's
promise that a seed plus a configuration is enough to recover a dataset.

The second run is a session-scoped fixture rather than a per-test call, so the whole
file costs two generations instead of five.

The negative control matters as much as the positive one: without it, a
``content_hash`` that always returned the same digest would pass the equality test
forever.
"""

from __future__ import annotations

import pytest

from simulator.config import config_fingerprint
from simulator.pipeline import content_hash, generate_dataset

ENTITY_TABLES = ("customers", "merchants", "accounts", "devices", "ips", "beneficiaries")


@pytest.fixture(scope="session")
def rerun(config_for):
    """A second, independent generation from an identically built configuration."""
    return generate_dataset(config_for())


@pytest.fixture(scope="session")
def other_seed(config_for):
    """Same configuration, different seed — the negative control."""
    return generate_dataset(config_for(seed=43))


@pytest.mark.slow
@pytest.mark.integration
def test_same_seed_and_config_reproduce_identical_rows(dataset, rerun):
    assert content_hash(rerun) == content_hash(dataset)
    assert len(rerun.transactions) == len(dataset.transactions)
    assert rerun.transactions == dataset.transactions


@pytest.mark.slow
@pytest.mark.integration
def test_same_seed_and_config_reproduce_the_ecosystem(dataset, rerun):
    assert rerun.ecosystem.entity_counts() == dataset.ecosystem.entity_counts()
    for table in ENTITY_TABLES:
        assert getattr(rerun.ecosystem, table) == getattr(dataset.ecosystem, table), (
            f"entity table {table!r} differs between two identical runs"
        )
    assert rerun.ecosystem.account_devices == dataset.ecosystem.account_devices
    assert rerun.ecosystem.account_ips == dataset.ecosystem.account_ips
    assert rerun.ecosystem.account_beneficiaries == (
        dataset.ecosystem.account_beneficiaries
    )
    assert rerun.ecosystem.customer_accounts == dataset.ecosystem.customer_accounts


@pytest.mark.slow
@pytest.mark.integration
def test_same_seed_and_config_reproduce_ground_truth_and_splits(dataset, rerun):
    assert len(rerun.ground_truth) == len(dataset.ground_truth)
    assert rerun.ground_truth.family_counts() == dataset.ground_truth.family_counts()
    assert (
        rerun.ground_truth.transaction_counts_by_family()
        == dataset.ground_truth.transaction_counts_by_family()
    )
    assert [r.attack_id for r in rerun.ground_truth.records] == [
        r.attack_id for r in dataset.ground_truth.records
    ]
    assert rerun.ground_truth.transaction_to_attack() == (
        dataset.ground_truth.transaction_to_attack()
    )
    assert rerun.boundaries.to_dict() == dataset.boundaries.to_dict()
    assert rerun.split_of == dataset.split_of


def test_manifest_hash_matches_its_own_rows(dataset):
    """The manifest is the artefact a reader trusts; it must describe these rows."""
    assert dataset.manifest["content_hash"] == content_hash(dataset)
    assert int(dataset.manifest["rng_streams"]) > 0


def test_config_fingerprint_is_stable_for_equal_configurations(config_for):
    assert config_fingerprint(config_for()) == config_fingerprint(config_for())


@pytest.mark.slow
def test_a_different_seed_produces_different_rows(dataset, other_seed):
    """Negative control: the hash must actually depend on the generated data."""
    assert content_hash(other_seed) != content_hash(dataset)
