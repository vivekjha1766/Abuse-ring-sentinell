# Checkpoint 1 Verification Report — Synthetic Data, Attacks & Ground Truth

**Status:** Checkpoint 1 complete and implementation-frozen (2026-08-25).
**Owner:** Data Agent (see `AGENTS.md`)

**Purpose.** Record what Checkpoint 1 actually delivers, what was measured, how it
was measured, and what remains unproven. Every number below was produced by
running the frozen simulator at seed 42; nothing is estimated, and anything not
measured is labelled **Not yet measured.** rather than filled in. This is a
verification record for a *production-style portfolio prototype*, not a
production readiness certificate for a real banking environment.

---

## 1. Scope

Checkpoint 1 covers the synthetic data layer only: the entity ecosystem, normal
transaction behaviour, seven parameterized attack families, the ground-truth
registry, the temporal split, the dataset-level validator, the adversarial
shortcut audit, the label quarantine, the regression test suite, and the single
dataset CLI the specification names.

**In scope and verified here.** Entity and transaction integrity; ground-truth
completeness; temporal split correctness; point-in-time causality; reproducibility
from seed and config; fraud prevalence at 1%, 5%, 10% and 25%; absence of oracle
shortcuts; the one-way label barrier.

**Explicitly out of scope.** No features are engineered, no model is trained, no
threshold is chosen, and no detection metric is reported. Feature construction and
the first models belong to Checkpoint 2; the ML-readiness audit report and its
accompanying CLI belong to Checkpoint 1.5. The held-out split has been generated
but never read for any purpose beyond counting its rows.

The dataset is synthetic throughout. No real personal data, no real merchant
names, no real card or account numbers, and no real IP ranges are used; the
currency code is `XTS`, the ISO 4217 code reserved for testing.

---

## 2. Files and modules implemented

**Simulator (frozen).**

| Module | Responsibility |
|---|---|
| `simulator/config.py` | `SimulationConfig`, per-split attack configuration, `default_config`, `config_fingerprint` |
| `simulator/base_generator.py` | `SeedRegistry` — named, independently seeded RNG streams; `TimeWindow`; ID minting |
| `simulator/entities.py` | six entity types, ecosystem assembly, device/IP/beneficiary sharing, `sharing_statistics`, `shuffle_entity_tables` |
| `simulator/transactions.py` | `Transaction`, `TRANSACTION_FIELDS`, normal behaviour (per-account profiles, seasonality, travel, legitimate outliers) |
| `simulator/ground_truth.py` | `AttackRecord`, `GroundTruthRegistry`, `ATTACK_FAMILIES`, `DETECTION_CLASS`, `validate_record_completeness`, JSON persistence |
| `simulator/attacks/` | seven family generators plus `base.py` (`AttackContext`, `attack_window`) |
| `simulator/pipeline.py` | ecosystem → normal traffic → boundaries → attack injection → manifest; `content_hash` |
| `simulator/validation.py` | `validate_dataset` — referential, temporal, causal, prevalence, ground-truth and non-separability checks |
| `simulator/leakage.py` | `LABEL_FIELDS`, `ENTITY_LABEL_FIELDS`, `FEATURE_SAFE_FIELDS`, `assert_feature_safe`, `feature_safe_row` |
| `simulator/shortcut_audit.py` | adversarial single-feature audit with the `oracle` / `strong` / `ok` / `quarantined` verdict scale |

**CLI.** `scripts/generate_dataset.py` — the only script the specification names.
`ROADMAP.md` lists it among the Checkpoint 1 artifacts; `DATA_DICTIONARY.md` §7
identifies it as how datasets are reproduced from a seed and config; `README.md`
places it in `scripts/`; `docs/DATA_SPLIT_PROTOCOL.md` §4 requires that it never
import the held-out loader. A grep of every Markdown file in the repository finds
no other script named anywhere in the specification. In particular there is no
requirement for a separate `scripts/audit_shortcuts.py`: the shortcut audit is
invoked from the regression suite today, and its standalone report is a
Checkpoint 1.5 deliverable. None was created, and this corrects an earlier
statement in the build log that two scripts were outstanding.

The CLI takes three options and no others: `--seed` (default 42), `--prevalence`
(default 0.05, restricted to the four supported values) and `--out` (default
`./data`). Seed and prevalence are read from the command line only, never from the
environment, so an exported variable cannot silently change what `make data`
produces. It writes:

```
data/raw/transactions.csv            all rows, canonical TRANSACTION_FIELDS order
data/raw/{customers,merchants,accounts,devices,ip_addresses,beneficiaries}.csv
data/raw/{account_devices,account_ips,account_beneficiaries,customer_accounts}.csv
data/raw/ground_truth.json           via GroundTruthRegistry.write_json
data/raw/manifest.json               config, fingerprint, boundaries, counts, content hash
data/raw/validation.json             the ValidationReport for this dataset
data/splits/{warmup,train,validation,heldout}/transactions.csv
```

Two implementation notes matter for verification. First, CSV cells are serialised
by delegating to the same private canonicaliser the manifest's `content_hash`
uses, rather than re-implementing its rules; this is what allows the stored
`transactions.csv` to be checked against the manifest instead of drifting from it
(§9). Second, warm-up traffic gets its own directory alongside the three
evaluation splits. It is not a fourth split — it is never scored — but
`docs/DATA_SPLIT_PROTOCOL.md` §5 exists so that 7-day rolling features have real
history on the first TRAIN day, and a feature builder can only obtain that history
without touching held-out rows if warm-up is reachable separately from
`data/raw/`. The script writes every split from the partition the pipeline already
computed in memory and reads nothing back, so it imports no loader of any kind.

**Tests.** `tests/simulator/` — ten files, one per coverage area plus `conftest.py`
(§13).

---

## 3. Dataset configuration

The primary configuration, seed 42, as generated on 2026-08-25:

| Property | Value |
|---|---|
| Master seed | 42 |
| Fraud prevalence (configured) | 0.05 — **primary synthetic benchmark configuration** |
| Evaluation window | 2026-01-01T00:00:00+00:00 → 2026-05-01T00:00:00+00:00 (120 days) |
| Warm-up | 14 days of normal-only history before the window |
| Currency | `XTS` (ISO 4217 test code) |
| Config fingerprint | `80c206b06d07d42b` |
| Transaction content hash | `9347662cdeda735ed22492a7de830ccf9f0c50ffab86df8c01992fc6c2e8a68f` |
| Named RNG streams issued | 289 |
| Transactions | 156,352 total — 7,054 fraud, 149,298 normal |
| Realised prevalence, evaluation window | 0.04951 |

5% is a benchmark choice made so that the same pipeline can be exercised across
four base rates. It is **not** a real-world or enterprise industry fraud rate, and
nothing in this project should be read as claiming otherwise.

**Entity counts.**

| Entity | Count | Notes |
|---|---|---|
| Customers | 1,000 | 100 synthetic identities (10.0%) |
| Merchants | 189 | 49 fake (25.9%), each with legitimate cover traffic |
| Accounts | 1,892 | mean 1.62 devices per account |
| Devices | 1,124 | 820 observed serving more than one account; 67 marked legitimately shared |
| IP addresses | 861 | 748 observed serving more than one account |
| Beneficiaries | 1,629 | |

Sharing is deliberately not fraud-exclusive: of the 820 devices seen serving more
than one account, 270 (32.9%) carry no fraudulent row at all. Normal behaviour is
non-trivial in the ways `ROADMAP.md` requires — heterogeneous per-account volumes
and amount profiles, hourly share ranging 0.0047–0.0738 and day-of-week share
0.1154–0.1726, legitimate travel, and legitimate large transfers (the largest
legitimate amount, 51,611.01, is an order of magnitude above the largest fraudulent
amount, 5,563.89). Fraud and legitimate amount distributions overlap heavily: the
overlap coefficient is 0.5292, 28,769 legitimate rows sit above the fraudulent
median, and the best single amount band reaches only F1 0.1949.

---

## 4. Prevalence configurations

All four supported prevalences were generated at seed 42 and validated. The
validator's prevalence check uses a **relative** tolerance of 0.30 against the
target, measured over the evaluation window only.

| Configured | Rows | Fraud rows | Realised (overall) | Attacks | Validation |
|---|---|---|---|---|---|
| 0.01 | 148,163 | 1,489 | 0.010050 | 53 | PASS — 0 errors, 0 warnings |
| 0.05 | 156,352 | 7,054 | 0.045116 | 279 | PASS — 0 errors, 0 warnings |
| 0.10 | 167,163 | 14,829 | 0.088710 | 608 | PASS — 0 errors, 1 warning |
| 0.25 | 209,073 | 44,263 | 0.211711 | 1,923 | PASS — 0 errors, 2 warnings |

Realised prevalence per split at 5%: train 0.04878, validation 0.05047, heldout
0.05077, warm-up 0.0 by construction. The overall figure (0.045116) sits below the
per-split figures because it includes the fraud-free warm-up period in its
denominator; the validator therefore measures the evaluation window, where the
realised value is 0.04951.

The two warnings are pre-existing, were judged not to be defects, and were left
unfixed under the freeze: `benign_sharing_rare` at 10% and 25%, and
`amount_informative` at 25%. Both are discussed in §14.

---

## 5. Attack families

Seven families, each parameterized rather than hard-coded, reported in two classes
because they are different detection problems and conflating them would flatter the
coordinated-abuse results.

**Class A — individual anomalies.** `transaction_spike`, `account_takeover`.
**Class B — coordinated abuse.** `fake_merchant_ring`, `shared_device_ring`,
`circular_money_flow`, `synthetic_identity_cluster`,
`microtransaction_attack`.

At 5%, seed 42 — 279 attack instances producing 7,054 fraudulent rows:

| Family | Class | Instances | Fraud rows |
|---|---|---|---|
| `transaction_spike` | A | 72 | 678 |
| `account_takeover` | A | 56 | 409 |
| `fake_merchant_ring` | B | 34 | 1,398 |
| `shared_device_ring` | B | 33 | 1,012 |
| `circular_money_flow` | B | 33 | 492 |
| `microtransaction_attack` | B | 32 | 1,893 |
| `synthetic_identity_cluster` | B | 19 | 1,172 |

Instance counts and row counts rank differently on purpose: one microtransaction
attack contributes hundreds of rows while one takeover contributes a handful, which
is why the prevalence budget is expressed in rows and the mix in proportions.

**Every family appears in every split.** `families_per_split` measures 7 for
train, validation and heldout. Two further checks guard against the families being
parameterized in name only: `families_with_constant_footprint` is empty (no family
produces the same shape every time), and
`families_with_identical_params_across_splits` is empty (no family carries
identical parameters from one split to the next), which is what
`docs/DATA_SPLIT_PROTOCOL.md` §6 requires so that detectors are tested against
attack *configurations* they have not seen.

---

## 6. Ground-truth integrity

`incomplete_ground_truth_records`: **0**. Every one of the 279 records names at
least one affected transaction, at least one affected entity, and the generation
parameters that produced it, and has `end_time >= start_time` — the four
conditions FR-3 requires for the record to support transaction-, entity-, ring-
and campaign-level evaluation.

Verified by the regression suite over the full registry:

- The set of transactions flagged `is_fraud` equals the union of all
  `affected_transaction_ids`, in both directions — no flagged row is unclaimed, no
  claimed row is unflagged.
- No transaction is claimed by two attacks.
- Every fraudulent row's `attack_id` resolves to a record whose family agrees with
  `transaction_to_attack()` and with the `by_id` index.
- Normal rows carry no `attack_id`.
- `family_counts()` and `transaction_counts_by_family()` reconcile with the rows
  themselves and with the manifest.
- Each record's `split` matches the split its window falls in, and each record's
  affected transactions all fall inside `[record.start_time, record.end_time]`.

The registry is written once by the simulator to `data/raw/ground_truth.json` and
is consumed only by evaluation. It is not part of the feature-safe view (§13).

One deliberate property worth stating plainly, because it looks like a defect and
is not: `entities_appearing_only_in_fraud` is non-zero (2 accounts, 129
beneficiaries, 114 devices, 13 IPs). Some attack-minted entities have no
legitimate history — that is realistic. What matters is that attacked accounts are
overwhelmingly embedded in normal traffic rather than isolated:
`attacked_accounts_also_in_normal_traffic` is 722 of 724, an embedding rate of
0.9972.

---

## 7. Temporal split integrity

Boundaries are the 60th and 80th percentiles of in-window *normal* transaction
timestamps, so each split holds close to its configured share of traffic
regardless of seasonality. Splits are half-open `[start, end)`; a row landing
exactly on a boundary falls into the **later** split.

**Boundary timestamps, seed 42, 5%:**

| Boundary | Timestamp |
|---|---|
| `evaluation_start` | 2026-01-01T00:00:00+00:00 |
| `train_end` | 2026-03-17T18:16:58.655506+00:00 |
| `validation_end` | 2026-04-09T16:45:44.105213+00:00 |
| `evaluation_end` | 2026-05-01T00:00:00+00:00 |

| Split | Transactions | Fraud | Normal | Realised prevalence |
|---|---|---|---|---|
| warm-up | 13,864 | 0 | 13,864 | 0.0 |
| train | 85,985 | 4,194 | 81,791 | 0.04878 |
| validation | 28,434 | 1,435 | 26,999 | 0.05047 |
| heldout | 28,069 | 1,425 | 26,644 | 0.05077 |

Realised split fractions over the evaluation window are 0.6035 / 0.1996 / 0.1970
against a configured 0.60 / 0.20 / 0.20. The residual comes from attack rows,
which are injected after the boundaries are fixed; the regression suite therefore
does not assert exact proportions, which are already covered by a warning-level
validator check.

Verified by the regression suite: every transaction is assigned exactly one split
and every split assignment matches `SplitBoundaries.split_for(timestamp)`; every
row's timestamp lies inside its split's window; the observed extremes are strictly
ordered, so the maximum train timestamp precedes the minimum validation timestamp
and the maximum validation timestamp precedes the minimum heldout timestamp; the
warm-up period contains zero fraud and no attack records; and every attack record
sits wholly inside one split's window, so no ring is cut in half by a boundary.

The three evaluation splits and the warm-up history are written to four separate
directories. Row identity was checked after writing: the union of the four split
files equals `data/raw/transactions.csv` exactly (156,352 rows), with no row in
two split files, and no held-out transaction id present in any other split
directory. The held-out split has not been read by anything except this row count.

---

## 8. Causality checks

No feature can be computed from information that did not exist yet, so no
transaction may reference an entity created after it. This is treated as an error,
not a warning, and it is measured for all six referenced entity types.

`causality_violations` at 5%, seed 42:

| Reference | Violations |
|---|---|
| transaction before account creation | 0 |
| transaction before customer creation | 0 |
| transaction before merchant creation | 0 |
| transaction before device creation | 0 |
| transaction before IP creation | 0 |
| transaction before beneficiary creation | 0 |

`entities_created_after_end` is 0 for all six types: no entity is minted beyond
the dataset's own horizon. `dangling_references` is empty: every referenced id
resolves. The regression suite additionally asserts the stronger set-inclusion
form — for every row, all entities it names are drawn from the population that
already existed at its timestamp, which catches a generator sampling from the
final entity pool even if per-field ages happened to come out non-negative — and
that no account predates the customer who owns it, and that no attack window
starts before the simulated history begins.

This check is not decorative. An earlier iteration of the generator produced
accounts with ages as negative as −108 days, and that regression is what the
parameterized test now pins.

Two things are deliberately **not** asserted, and the omissions are documented in
the test module rather than left silent. Device and IP *adoption* timing is not
asserted, because attack generators emit deliberate cover traffic on freshly
minted devices, so the rule is not uniform across rows and asserting it would
manufacture a failure. Ecosystem link-table timing is not asserted, because the
link tables record relationships as of the end of generation, not a history, so
there is no per-link timestamp to compare.

---

## 9. Reproducibility

**Byte-identical output from the same seed and config — verified.** The CLI was
run twice at seed 42 and prevalence 0.05, into two different output directories,
and all **18 generated files** compared byte-for-byte identical: `transactions.csv`,
the six entity tables, the four link tables, `ground_truth.json`, `manifest.json`,
`validation.json`, and the four split files.

**The stored dataset verifies against its own manifest.** Re-joining the cells of
`data/raw/transactions.csv` with `|` and hashing the result reproduces the
manifest's `content_hash` exactly
(`9347662cdeda735ed22492a7de830ccf9f0c50ffab86df8c01992fc6c2e8a68f`) over all
156,352 rows. This is a consequence of the CLI delegating to the pipeline's own
canonicaliser rather than inventing a second serialisation, and it means a
downstream consumer can check the file it received without regenerating anything.

**In-process reproducibility — verified by the regression suite.** Two generations
from the same config produce identical transaction rows, identical ecosystem
tables, identical ground truth, identical boundaries and identical split
assignments; the manifest's recorded `content_hash` equals a freshly computed one;
and `config_fingerprint` is stable across runs. Because `content_hash` covers
transaction rows only — including the label columns, deliberately, since the same
rows with different labels are not the same dataset — ecosystem and registry
equality are asserted separately rather than assumed from the hash.

**Negative control.** Seed 43 produces a different `content_hash`, confirming the
comparison has power and is not trivially passing.

Determinism comes from 289 named, independently seeded RNG streams derived from
the single master seed, so adding a generator step cannot perturb the streams of
unrelated components.

---

## 10. Shortcut-audit results

The audit sweeps every candidate single-feature leak — amount and amount ratios,
velocity and burst counts, account age, geography, entity creation time, sharing
structure and degree, merchant profile, generator metadata, and the quarantined
label fields — against every attack family and against `any_fraud`. Its gates are
`ORACLE_AUC = 0.98`, `STRONG_AUC = 0.90`, `ORACLE_PRECISION = 0.95` and
`ORACLE_RECALL = 0.50`. A feature at or above the oracle gate fails the audit
outright: it would mean the generator, not the modelling, was producing the
metrics.

Freshly measured at seed 42 on 2026-08-25:

| Prevalence | Shortcut tests | Oracles | Strong | `ok` | Quarantined | Rows audited | Result |
|---|---|---|---|---|---|---|---|
| 0.01 | 475 | **0** | 17 | 434 | 24 | 134,648 | PASS |
| 0.05 | 476 | **0** | 14 | 438 | 24 | 142,488 | PASS |
| 0.10 | 476 | **0** | 12 | 440 | 24 | 152,794 | PASS |
| 0.25 | 476 | **0** | 13 | 439 | 24 | 193,029 | PASS |

The highest non-label AUC observed is 0.9682 (`account_burst_count_1h` →
`transaction_spike`, at 1%), comfortably below the 0.98 gate. The strongest
signals at 5% are `account_burst_count_1h` → `transaction_spike` (0.9472),
`amount_over_merchant_ticket` → `fake_merchant_ring` (0.9325),
`distance_from_home_km` → `account_takeover` (0.9251),
`amount_over_account_baseline` → `microtransaction_attack` (0.9215) and `amount` →
`circular_money_flow` (0.9211).

**Strong signals are wanted, not tolerated.** A dataset in which no single feature
carried real signal would be undetectable by construction and useless for the
modelling checkpoints. The verdict scale encodes exactly this: `strong` passes and
is reported, `oracle` fails. The audit's purpose is to catch the generator
accidentally stamping the answer on the row, not to flatten the data until nothing
is learnable.

The 24 `quarantined` results per prevalence are the label fields, probed on
purpose. At 5% `customer.is_synthetic_identity` predicts
`synthetic_identity_cluster` at AUC 0.9903 and `merchant.is_fake` predicts
`fake_merchant_ring` at 0.9301 — as they must, being the answer key. They are
recorded rather than counted as failures, and measuring them is what justifies the
quarantine instead of leaving it an unexamined assumption.

One caveat on reading the audit's precision and recall columns: the band search
sweeps 60 *index* quantile cuts, so on a tied integer-valued feature the reported
precision and recall describe an index slice of the tied population rather than
the value rule printed alongside them. AUC is the figure the verdict is based on,
and it is the figure to trust.

---

## 11. The IP oracle: discovered and remediated

**Symptom.** At 1% and 25% prevalence the audit reported `ip_account_degree` →
`account_takeover` above the oracle gate.

**Root cause.** Not novelty — uniqueness. Normal traffic is generated before
attacks, so any IP an attack mints can never accumulate organic legitimate
traffic, and its account degree is 1 by construction. The takeover generator
minted one dedicated IP per victim, which made "degree exactly 1" a near-perfect
identifier of takeover rows. This is the same defect pattern that had already been
found and fixed several times during Checkpoint 1: entity creation-order leakage,
fake merchants without cover traffic, a static ecosystem, absent legitimate travel
and large legitimate transfers, ring beneficiary concentration, and a fixed 12-hour
device freshness.

**Fix.** The smallest possible generator-side change, and the only generator
correction accepted at freeze time: takeover origin IPs are now drawn from four
realistic origins rather than always minted fresh — a previously seen
household or shared IP, a shared-infrastructure IP, or a new dedicated IP — via
new `ip_origin_weights` configuration. Two files changed:
`simulator/config.py` and `simulator/attacks/account_takeover.py`. The feature was
not removed, the audit was not weakened, and no threshold was moved.

**Result.** `ip_account_degree` → `account_takeover` no longer appears among the
oracle findings at any prevalence. `ip_account_degree` still carries legitimate
signal elsewhere — 0.9156 against `synthetic_identity_cluster` at 10% and 0.9227
at 25%, both `strong` — which is the intended outcome: the artifact is gone, the
realistic signal remains.

---

## 12. `device_account_degree`: retained as a non-oracle signal

The pre-freeze audit flagged `device_account_degree` → `account_takeover` at AUC
0.9010 at 5%, close enough to the gate to warrant asking whether it was a second
instance of the same artifact. It was measured at all four prevalences and
inspected rather than reflexively fixed.

| Prevalence | AUC | Verdict | Band rule | Precision | Recall |
|---|---|---|---|---|---|
| 0.01 | 0.9554 | strong | `1 <= x <= 1` | 0.0136 | 0.8108 |
| 0.05 | 0.9010 | strong | `1 <= x <= 1` | 0.0448 | 0.7013 |
| 0.10 | 0.8006 | ok | `1 <= x <= 1` | 0.0963 | 0.3463 |
| 0.25 | 0.6286 | ok | `1 <= x <= 1` | 0.1818 | 0.5327 |

**It does not cross the oracle gate at any prevalence**, and it falls below the
strong threshold entirely at 10% and 25%. Two things distinguish it from the IP
oracle. It decays steeply as base rate rises, which an artifact of construction
would not: a mechanical stamp stays perfect regardless of how many attacks are
injected. And the implied rule has almost no precision — 0.0448 at 5% — because
single-account devices are the overwhelming majority of devices in a realistic
ecosystem, so "degree 1" cannot be acted on the way an oracle could.

The signal reflects a behaviour the takeover family is *supposed* to have:
takeovers frequently occur from a device the victim's account has not used before.
Removing it would mean asking the takeover family to stop behaving like a
takeover. It is therefore **intentionally retained as a non-oracle strong signal**
and is frozen. The regression suite pins both failure directions: that the probe
is still being taken (so nobody could miss a future drift upward) and that it
remains below the oracle gate. Its magnitude below that gate is deliberately not
constrained.

---

## 13. Regression test suite

Ten files under `tests/simulator/`, 1,126 lines, covering the nine areas
mandated for the freeze: deterministic reproducibility, entity integrity,
transaction integrity, ground-truth integrity, temporal split integrity,
point-in-time causality, prevalence configuration at all four base rates,
shortcut-audit regression, and label/quarantine safety.

| Metric | Value |
|---|---|
| Tests collected | 106 |
| Passed | 106 |
| Failed | 0 |
| Skipped | 0 |
| Production files modified to make tests pass | 0 |

> **How these tests were executed — read this before citing the result.**
> The 106 tests were executed with a **local pytest-compatible shim**, not with
> real pytest. The development workspace is network-isolated and PyPI is
> unreachable from it (`pip install pytest` fails with a proxy 403), so pytest
> could not be installed. The shim is a small scratch harness implementing only
> the API these tests use — `fixture(scope=…)`, `mark` including `parametrize`,
> `raises`, `approx`, `skip`/`fail` — plus a runner that loads `conftest.py`,
> resolves fixtures with session caching, expands parametrized cases, and reports
> collected / passed / failed / skipped counts. It lives outside the repository
> and is not committed.
>
> **Real pytest execution remains pending.** The tests are written against
> pytest 8 and the project's `[tool.pytest.ini_options]` configuration
> (`--strict-markers`, `--strict-config`, `filterwarnings = ["error::DeprecationWarning"]`,
> markers `slow` / `integration` / `leakage`), but that configuration has never
> been applied to them, and strict-marker and strict-config enforcement in
> particular has not been exercised. The `ROADMAP.md` Checkpoint 1 exit criterion
> "`uv run pytest` green" is therefore **not yet satisfied**, and must be run in a
> network-connected environment before the suite result is quoted anywhere as a
> pytest result.

Suite conventions worth preserving. Datasets are memoised in session-scoped
fixtures keyed by `(prevalence, seed)` and audit reports are memoised per
prevalence, because one generation costs 12–40 seconds depending on prevalence and
a generate-plus-audit cycle 40–78 seconds; without memoisation the prevalence sweep
and the audit regression would generate the same four datasets twice over. The
audit gate constants, `LABEL_FIELDS`,
`ENTITY_LABEL_FIELDS` and `SUPPORTED_PREVALENCES` are pinned as freeze guards, so
the suite cannot be satisfied by moving the goalposts rather than fixing the data
— the one failure mode a regression test for an audit cannot detect from its own
results. Invariants that are deliberately not asserted are documented in the
module docstring that omits them, with the reason.

The label-safety file is a negative-control test as much as a positive one. Each
quarantined name is fed to `assert_feature_safe` bare and in the dotted and
underscore-prefixed forms a real join produces, hidden among legitimate columns,
and as the whole raw schema, and each must raise `LeakageError` — because a test
that only checked the safe path would pass just as well against a barrier that
never fired. `feature_safe_row` is asserted to *omit* the labels rather than blank
them, so a downstream typo raises `KeyError` instead of quietly reading `None`.

Execution was chunked around a 180-second shell timeout in the development
workspace: the five integrity files share one 5% generation (15.5 s for 58 tests),
label quarantine 44.4 s for 15, reproducibility 45.2 s for 6, the prevalence sweep
14.9 / 18.1 / 21.4 / 46.1 s, and the audit regression 49.5 / 56.7 / 70.0 / 107.4 s
at 1 / 5 / 10 / 25%. The aggregate across 12 invocations was 489 s only because
each chunk re-pays generation; a single in-process run would be roughly five to
six minutes.

---

## 14. Known limitations

**Real pytest has never run against this suite.** See §13. This is the most
significant open item in Checkpoint 1.

**`make data` has not been exercised as written.** The Makefile target is
`uv run python scripts/generate_dataset.py`. `uv run` cannot function in the
network-isolated workspace — it attempts to download a CPython interpreter and
fails at the proxy — so the script was executed directly with the system CPython
3.10.12 instead. The script itself bootstraps `sys.path` from its own location and
imports cleanly with no `PYTHONPATH` set, so the `uv run` form is expected to work
unchanged; expected is not verified.

**Two validator warnings are open and were consciously not fixed.**
`benign_sharing_rare` fires when fewer than 20% of multi-account devices are
fraud-free: measured 32.9% at 5% prevalence (no warning), 15.2% at 10% and 3.2% at
25%. The check is arguably mis-specified rather than the data being wrong — as the
base rate rises, the chance that *any* given shared device escapes carrying a
fraudulent row falls mechanically, so the threshold cannot hold at 25% no matter
how realistic the sharing structure is. The related error condition, "every
multi-account device carries fraud", never fires. `amount_informative` fires at 25%
with "best amount band reaches F1=0.422"; for context, a classifier that flagged
every row would reach `2p/(1+p) ≈ 0.349` at the realised 21.2% base rate, so the
band is informative but not dramatically so. Both are warning-level, neither fails
validation, and both are Checkpoint 1.5 questions.

**Realised prevalence undershoots the target overall.** 0.045116 against a
configured 0.05 at 5%, and 0.211711 against 0.25 — inside the 0.30 relative
tolerance, and expected: the overall figure includes the fraud-free warm-up period
in its denominator, and the last attack injected overshoots or undershoots the row
budget by however many transactions it happens to contain. The per-window and
per-split figures land close to target.

**Split proportions are approximate.** 0.6035 / 0.1996 / 0.1970 against
0.60 / 0.20 / 0.20, because boundaries are fixed on normal traffic before attacks
are injected.

**The audit is single-feature by construction.** It cannot detect a shortcut that
only exists in a *combination* of features, and it tests separability with band
rules and AUC rather than with a trained model. A model-based leakage check
belongs to Checkpoint 1.5.

**No detection metric exists yet.** No features, no model, no threshold, no
precision/recall/AUC for any detector: **Not yet measured.** The results tables in
`EVALUATION.md`, `docs/MODEL_BASELINES.md` and `MODEL_CARD.md` remain unfilled and
should stay that way until the checkpoint that measures them.

**Synthetic data has synthetic limits.** Behaviour is drawn from parameterized
distributions chosen for plausibility, not fitted to any real payment population.
Absolute numbers from this dataset describe this dataset. They do not transfer to a
real portfolio, and no claim in this project should be read as evidence about real
fraud rates or real detector performance.

---

## 15. Checkpoint 1 status

**Complete and implementation-frozen as of 2026-08-25**, with one exit criterion
carried forward.

| Exit criterion (`ROADMAP.md`) | Status |
|---|---|
| Six entity types with stable IDs and creation timestamps | ✅ verified (§3, §6) |
| Normal behaviour non-trivial | ✅ verified (§3) |
| All seven attacks parameterized, no hard-coded shapes | ✅ verified (§5) |
| Ground truth records all required fields | ✅ verified — 0 incomplete records (§6) |
| Same seed + config reproduces byte-identical output | ✅ verified — 18/18 files (§9) |
| Configurable prevalence at 1 / 5 / 10 / 25% | ✅ verified (§4) |
| Temporal 60/20/20 split, differing attack configs per split | ✅ verified (§7, §5) |
| `uv run pytest` green | ⏳ **pending** — 106/106 pass under a local shim; real pytest not yet run (§13) |
| Zero oracle shortcuts at all four prevalences | ✅ verified (§10) |

The simulator, its configuration, the attack generators, the audit thresholds and
the labels are frozen. `device_account_degree` is retained deliberately (§12). The
next checkpoint is 1.5 (ML-readiness audit), which is not begun.

---

## Commands executed

Run from the repository root with the system CPython 3.10.12:

```bash
python scripts/generate_dataset.py                      # data/, seed 42, prevalence 0.05
python scripts/generate_dataset.py --out /tmp/data2     # second run, determinism check
diff -r --exclude=.gitkeep data /tmp/data2              # byte comparison — no file differs
```

The prevalence sweep (§4), the four-prevalence shortcut audit (§10), the
`transactions.csv` → `content_hash` re-derivation and the split-partition identity
check (§7) were run as short scratch scripts outside the repository, calling only
`default_config`, `generate_dataset`, `validate_dataset` and `audit_shortcuts`
from the frozen simulator. They are diagnostics, not deliverables, and were not
committed. The 106 regression tests were executed with the local
pytest-compatible shim described in §13; **real pytest was not run.**

---

## Related documents

- `ROADMAP.md` — Checkpoint 1 artifacts and exit criteria
- `DATA_DICTIONARY.md` — entity and transaction schemas, data lifecycle
- `docs/DATA_SPLIT_PROTOCOL.md` — split rules and the held-out seal
- `docs/FEATURE_LEAKAGE_POLICY.md` — point-in-time safety and the label quarantine
- `docs/ML_READINESS.md` — the Checkpoint 1.5 audit this report feeds
- `CONTRIBUTING.md` — the non-negotiables this report is written under
