"""2. Entity integrity — identifiers are unique and every reference resolves.

Two distinct failure modes are covered. *Uniqueness*: the id minter must never issue
the same identifier twice, or two different customers silently become one entity in
any groupby a feature pipeline performs. *Referential closure*: every id appearing on
a transaction row, or in one of the ecosystem's link tables, must name an entity that
actually exists in its table — a dangling reference would surface downstream as a
missing join key rather than as an error here.

:func:`simulator.validation.validate_dataset` also checks referential integrity; this
file asserts it directly so a failure names the offending table and id instead of a
report code.
"""

from __future__ import annotations

import pytest

ID_FIELDS = (
    ("customers", "customer_id"),
    ("merchants", "merchant_id"),
    ("accounts", "account_id"),
    ("devices", "device_id"),
    ("ips", "ip_id"),
    ("beneficiaries", "beneficiary_id"),
)


@pytest.mark.parametrize("table,id_field", ID_FIELDS)
def test_entity_ids_are_unique(dataset, table, id_field):
    rows = getattr(dataset.ecosystem, table)
    ids = [getattr(row, id_field) for row in rows]

    assert len(ids) == len(set(ids)), f"{table} contains duplicate {id_field} values"
    assert all(isinstance(i, str) and i for i in ids)


@pytest.mark.parametrize("table,id_field", ID_FIELDS)
def test_entity_counts_match_the_tables(dataset, table, id_field):
    counts = dataset.ecosystem.entity_counts()

    assert counts[table] == len(getattr(dataset.ecosystem, table))


def test_accounts_reference_existing_customers(dataset):
    eco = dataset.ecosystem
    customers = set(eco.customer_by_id)

    dangling = {a.account_id for a in eco.accounts if a.customer_id not in customers}
    assert not dangling, f"accounts referencing unknown customers: {sorted(dangling)[:5]}"


def test_transactions_reference_existing_entities(dataset):
    """Every id on every row must resolve, including the optional ones."""
    eco = dataset.ecosystem
    lookups = {
        "account_id": set(eco.account_by_id),
        "customer_id": set(eco.customer_by_id),
        "merchant_id": set(eco.merchant_by_id),
        "device_id": set(eco.device_by_id),
        "ip_id": set(eco.ip_by_id),
        "beneficiary_id": set(eco.beneficiary_by_id),
    }

    dangling = {}
    for txn in dataset.transactions:
        for field, known in lookups.items():
            value = getattr(txn, field)
            if value is not None and value not in known:
                dangling.setdefault(field, []).append((txn.transaction_id, value))

    assert not dangling, f"transactions reference unknown entities: { {k: v[:3] for k, v in dangling.items()} }"


def test_transaction_account_and_customer_agree(dataset):
    """A row's ``customer_id`` must be the owner of its ``account_id``."""
    accounts = dataset.ecosystem.account_by_id

    mismatched = [
        txn.transaction_id
        for txn in dataset.transactions
        if accounts[txn.account_id].customer_id != txn.customer_id
    ]
    assert not mismatched, f"customer/account mismatch on rows {mismatched[:5]}"


def test_ecosystem_link_tables_reference_existing_entities(dataset):
    eco = dataset.ecosystem
    accounts = set(eco.account_by_id)
    customers = set(eco.customer_by_id)

    for table, known in (
        (eco.account_devices, set(eco.device_by_id)),
        (eco.account_ips, set(eco.ip_by_id)),
        (eco.account_beneficiaries, set(eco.beneficiary_by_id)),
    ):
        assert set(table) <= accounts
        for account_id, linked in table.items():
            unknown = set(linked) - known
            assert not unknown, f"account {account_id} links to unknown {sorted(unknown)[:3]}"

    assert set(eco.customer_accounts) <= customers
    for customer_id, owned in eco.customer_accounts.items():
        unknown = set(owned) - accounts
        assert not unknown, f"customer {customer_id} owns unknown accounts {sorted(unknown)[:3]}"
