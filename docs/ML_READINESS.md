# ML Readiness Audit

**Status:** Checkpoint 1.5 — **measured and PASSED** at 1%, 5%, 10%, and 25%.
**Owner:** Data + ML Agents (see `AGENTS.md`)

**Purpose.** This is a gate. It runs after the synthetic dataset exists
(Checkpoint 1) and before any feature or model work (Checkpoint 2). It exists to
prove the dataset is genuinely suitable for honest machine learning *before*
effort is spent modelling it.

**How to reproduce every number below.** One command per prevalence:

```
python scripts/audit_ml_readiness.py --prevalence 0.05 --json artifacts/reports/ml_readiness_p05.json
```

The script composes the Checkpoint 1 checks (`simulator/validation.py`,
`simulator/shortcut_audit.py`) with the seven checks this gate adds, and exits
non-zero if any criterion fails. Unless a cell says otherwise, every figure in
this document is the 5% run — the primary synthetic benchmark configuration — at
seed 42.

| Prevalence | Transactions | Fraud rows | Attacks | Config fingerprint | Gate |
|---|---|---|---|---|---|
| 1% | 148,163 | 1,489 | 53 | `b57b7ace246d04ac` | PASS |
| **5% (primary)** | **156,352** | **7,054** | **279** | **`80c206b06d07d42b`** | **PASS** |
| 10% | 167,163 | 14,829 | 608 | `f5d4d89ea88fca51` | PASS |
| 25% | 209,073 | 44,263 | 1,923 | `cd770f7cc183d65c` | PASS |

Two warnings recur and do not fail the gate: `benign_sharing_rare` at 10% and
25%, and `amount_informative` at 25%. Both are consequences of raising prevalence
against a fixed legitimate population — at 25% one row in four is fraudulent, so
benign device overlap becomes scarce relative to ring overlap. They are recorded
rather than suppressed, and 5% remains the configuration all headline numbers are
reported at.

---

## 1. Why this gate exists

A synthetic generator can accidentally make fraud trivially separable — for
example by giving every fraudulent transaction a suspiciously round amount, an
identical timestamp, or a unique device. A model trained on such data would post
excellent metrics that mean nothing, and the flaw might not be discovered until
much later. Catching a degenerate dataset after building four detection layers
wastes the entire effort.

The readiness audit is therefore adversarial toward our own data generator. Its
job is to *try to break* the dataset's credibility and confirm it survives. If a
check fails, the correct response is to **fix the generator, not the metric**.

---

## 2. Dataset sanity audit

Basic integrity before anything else.

| Check | Criterion | Status |
|---|---|---|
| Row counts | Transactions and entities match configured targets | PASS — 156,352 transactions; 189 merchants, 1,629 beneficiaries, 100 synthetic customers |
| ID uniqueness | Every entity ID is unique and stable across the run | PASS — 156,352 unique transaction IDs for 156,352 rows |
| ID stability under reseed-same | Same seed reproduces identical IDs | PASS — two independent audit runs at seed 42 produced byte-identical manifests (`content_hash 9347662c…`) |
| Timestamp validity | All timestamps parseable, single timezone, none in the future relative to the dataset clock | PASS — all timezone-aware UTC; 0 entities created after the window end |
| Timestamp ordering | Transactions can be totally ordered; ties handled deterministically | PASS — 210 exact timestamp ties, all ordered by `(timestamp, transaction_id)` |
| Duplicate transactions | No exact/near-duplicate rows (esp. across split boundaries) | PASS — 0 exact duplicates, 0 near-duplicates (same account, merchant, amount, minute) |
| Null policy | No nulls in non-nullable fields | PASS — declared types and vocabularies hold for every row |
| Referential integrity | Every transaction references existing customer/account/merchant/device/IP/beneficiary | PASS — 0 dangling references |
| Label prevalence | Fraud rate matches the configured value (e.g. 5%) within tolerance | PASS — 0.04951 realised against 0.05 configured |

---

## 3. Triviality audit (the important one)

The dataset must not be solvable by a single threshold. This section actively
tries the naive rules and confirms they are weak. 476 probes are run at 5%, 10%,
and 25%, and 475 at 1%: the audit refuses to score a probe with fewer than 10
positive rows carrying a usable feature value, and at 1% prevalence
`account_beneficiary_degree → synthetic_identity_cluster` falls below that floor.
Each probe scores one observable feature against `any_fraud` and against each of
the seven attack families.

| Check | Criterion | Status |
|---|---|---|
| "Large amount ⇒ fraud" | This rule is a **weak** classifier (poor precision/recall) | PASS — amount → any_fraud AUC 0.598; best band `156.8 ≤ x ≤ 2256` reaches F1 0.206 |
| "High transaction count ⇒ fraud" | Also **weak** | PASS — account transaction count AUC 0.722 / F1 0.196; 1-hour burst count AUC 0.636 / F1 0.223 |
| Legitimate high-value outliers exist | Some genuinely large legitimate transactions are present | PASS — largest legitimate amount 51,611 against a fraud maximum of 5,564; 28,769 legitimate rows exceed the fraud median |
| Legitimate device sharing exists | Households / shared machines create benign device overlap | PASS — 820 devices used by more than one account, 270 of them carrying no fraudulent transaction (32.9%) |
| Legitimate IP sharing exists | NAT / CGNAT / shared networks create benign IP overlap | PASS — 748 IPs used by more than one account, 140 of them touched by no account that ever transacted fraudulently |
| No single feature perfectly separates | No feature yields a clean threshold split of the classes | PASS — 0 oracle findings at all four prevalences; strongest permitted single feature is entity reputation at AUC 0.800 |
| No giveaway artifact | Fraud rows do NOT share a tell (round amounts, identical timestamps, a unique device pattern, sequential IDs) | PASS — account creation order AUC 0.513, `account_unseen_in_train` AUC 0.508, amount distributions overlap at coefficient 0.529 |

Fourteen probes at 5% are classified **strong** rather than **oracle** — a family
is partly characterised by an observable, which is realistic and intended. The
clearest is `device_account_degree → account_takeover` at AUC 0.955 (1%), 0.901
(5%), 0.801 (10%), 0.629 (25%): shared-device pressure genuinely accompanies
takeover, it does not define it, and its collapse as prevalence rises is what
distinguishes a signal from a tell. These are recorded as a baseline for
Checkpoint 2, not fixed.

The highest AUC anywhere in the audit is `customer.is_synthetic_identity →
synthetic_identity_cluster` at 0.990, and it is reported as **quarantined**, not
as an oracle: it is a generator internal, classified `FORBIDDEN` in
`ml/feature_contract.py`, and `assert_model_input_safe` refuses it under every
spelling it could acquire through a join. The audit measures it precisely so the
cost of ever un-forbidding it is on the record.

If any naive rule turns out to be a *strong* classifier of `any_fraud`, the
generator is leaking the label into an observable feature and must be corrected.
The intent, restated from the PRD: neither "large amount = fraud" nor "many
transactions = fraud" may be universally true.

---

## 4. Class balance and prevalence

Prevalence is configurable at 1%, 5%, 10%, and 25%, with 5% the primary synthetic
benchmark configuration.

| Check | Criterion | Status |
|---|---|---|
| Configured prevalence honoured | Realised fraud rate ≈ configured, per split | PASS — train 0.04878, validation 0.05047, held-out 0.05077 |
| Prevalence sweep sane | 1/5/10/25% configs each produce coherent datasets | PASS — all four gate PASS; realised rates track configured at every setting |
| Both classes present in every split | No split is single-class | PASS — train 4,194/81,791, validation 1,435/26,999, held-out 1,425/26,644 (fraud/normal) |

---

## 5. Temporal distribution

Fraud must be woven through time, not parked in one block.

| Check | Criterion | Status |
|---|---|---|
| Fraud not time-confined | Fraud is not confined to one contiguous interval | PASS — fraud on 117 of 120 evaluation days; longest fraud-free gap 2 days; busiest day holds 2.6% of all fraud |
| Attacks in all splits | Each attack family appears in train, validation, and held-out | PASS — 7 of 7 families in each of the three splits |
| Configuration drift across splits | Attack parameters differ across splits per the generalization design | PASS — 0 families with identical parameters across splits; 0 with a constant footprint |
| Warm-up sufficiency | Enough pre-window history for 7-day features | PASS — 14 days configured, 13,864 rows spanning 13 days, 0 fraudulent; 78.8% of TRAIN accounts have warm-up history |
| Seasonality present | Hour-of-day / day-of-week structure exists in normal traffic | PASS — hourly share ranges 0.47%–7.38%, day-of-week 11.5%–17.3% |

Warm-up sufficiency is measured as *span plus coverage*, not span alone. A 14-day
window that contained no rows for an account would still leave that account's
7-day features starting cold, so the share of TRAIN accounts with any prior
history is reported alongside the span.

---

## 6. Leakage pre-checks and ground-truth quarantine

Verified before features are written, so the policy starts clean.

| Check | Criterion | Status |
|---|---|---|
| Ground-truth quarantine | No feature/model/scoring module imports the ground-truth registry | PASS — AST scan of 5 modules across `ml/`, `graph/`, `backend/`; 0 imports of `simulator.ground_truth`, 0 reads of `is_fraud` or `attack_id` |
| No future columns | No dataset column encodes a future-derived value | PASS — all 17 transaction columns classified in `ml/feature_contract.py`; `status` is aggregate-only, 0 causality violations of any kind |
| Split isolation | No row appears in more than one split | PASS — split counts sum exactly to 156,352; boundaries ordered and non-overlapping |
| Reputation source | Reputation inputs are historical confirmed outcomes only | PASS by construction — no reputation feature exists yet, and the only sanctioned way to build one is `HistoryView`, which cannot read at or beyond its as-of instant |
| Point-in-time safety at scale | Adding later transactions cannot change an earlier row's features | PASS — 4,000 sampled rows recomputed against a history truncated at the pivot: 0 feature mismatches, 0 boundary violations, 0 self-inclusions, 0 non-finite values |

The at-scale point-in-time check is the same experiment the unit tests run,
inverted. It builds the history twice — once over all 156,352 rows and once over
only the rows before a pivot instant — and recomputes velocity, volume,
deviation, previous-transaction and geographic-distance features for 4,000
sampled rows that precede the pivot. Any difference between the two would mean a
later transaction reached backwards. The dataset contains 210 pairs of rows
sharing an exact timestamp, so the strictly-less-than tie rule is exercised by
real data and not only by the fixtures.

Full policy in `docs/FEATURE_LEAKAGE_POLICY.md`; the enforcing tests are in
`tests/leakage/`.

---

## 7. Determinism and reproducibility

| Check | Criterion | Status |
|---|---|---|
| Seed reproducibility | Same seed + config → byte-identical dataset | PASS — two audit runs at seed 42 / 5% produced identical `config_fingerprint` and `content_hash`; confirmed independently by `tests/simulator/test_reproducibility.py` over rows, ecosystem, ground truth, and splits |
| Seed sensitivity | Different seed → different but statistically similar dataset | PASS — `test_a_different_seed_produces_different_rows` (seed 43) yields a different `content_hash` |
| Config capture | The generating config and seed are recorded with the dataset | PASS — the manifest records the full config, `config_fingerprint`, `content_hash`, seed, and RNG stream count (289 at 5%) |

---

## 8. Graph readiness

The graph layer needs a graph worth analysing. Measured over the projection a
Checkpoint 3 detector would actually see: edges derived from observed
transactions (account–device, account–IP, account–merchant, account–beneficiary)
plus beneficiary-to-linked-account links, never from the generator's private
relationship pools.

| Check | Criterion | Status |
|---|---|---|
| Connectivity | Entities are actually linked; the graph is not mostly isolated nodes | PASS — 5,481 nodes, 77,249 edges, 2 components, largest holds 99.96%, 0 isolated nodes |
| Degree distribution | Non-degenerate (not all-equal, not a single hub) | PASS — account degree min 1, median 34, p90 87, max 190, across 137 distinct values |
| Legitimate-sharing baseline | A measurable baseline of benign sharing exists to contrast against rings | PASS — 147 devices and 140 IPs shared by multiple accounts, none of which has ever transacted fraudulently (491/476 at 1%, 49/36 at 10%, 6/8 at 25%) |
| Rings not trivially isolated | Attack rings are embedded in the legitimate graph, not disconnected components a filter would trivially separate | PASS — 0 of 151 multi-account coordinated rings share entities exclusively with ring members |
| Cycle presence | Circular-money-flow attacks actually produce detectable cycles | PASS — 33 of 33 `circular_money_flow` attacks close a cycle in the account→beneficiary→account graph; 114 accounts lie on one |

Connectivity is computed with union-find and cycles with an iterative
three-colour DFS, both hand-rolled: NetworkX is a Checkpoint 3 dependency and
this gate must run before it is installed. The DFS is iterative rather than
recursive because a readiness audit that dies of a `RecursionError` is
indistinguishable from one that found nothing.

The benign-sharing figures here (147 devices) are stricter than §3's (270) on
purpose. §3 counts multi-account devices on which no fraudulent transaction
occurred; §8 counts multi-account devices no fraudulent *account* has ever
touched. The stricter definition is the one that matters to a graph filter,
because a connectivity heuristic sees a device's account set and not which of its
transactions were fraudulent.

---

## 9. The gate

The audit passes only if every criterion above passes.

| Section | Gate | Status |
|---|---|---|
| Dataset sanity | All integrity checks pass | PASS at 1/5/10/25% |
| Triviality | All naive rules are weak; no giveaway artifact | PASS — 0 oracles in 475–476 probes per prevalence |
| Class balance | Prevalence honoured, both classes per split | PASS at 1/5/10/25% |
| Temporal | Attacks spread across time and splits | PASS — 7/7 families per split, fraud on 117/120 days |
| Leakage pre-checks | Quarantine intact, splits isolated | PASS — 0 quarantine breaches, 0 point-in-time contaminations in 4,000 sampled rows |
| Determinism | Seed reproducibility confirmed | PASS — identical manifests across runs |
| Graph readiness | Connected, non-degenerate, rings embedded | PASS — 99.96% in one component, 0 isolated rings, 33/33 cycles |
| **Overall** | **All sections pass** | **PASS — Checkpoint 2 may begin** |

Not measured here, because it requires a model that does not exist yet: PR-AUC,
ROC-AUC, precision, recall, F1, and financial error cost are all **not yet
measured**. No threshold has been selected. The held-out split's row and label
counts appear above because the class-presence and prevalence checks read them as
dataset properties; nothing has been fitted, scored, or tuned on held-out, and the
sealed loader has never been called (see `docs/DATA_SPLIT_PROTOCOL.md` §4).

---

## 10. If the audit fails

Fix the data, not the scoreboard. Concretely:

- A naive rule is too strong → the generator is leaking the label into a feature;
  adjust the generation so fraud is not marked by an observable tell.
- Fraud is time-confined → distribute attacks across the timeline and all splits.
- Rings are trivially separable → embed rings in the legitimate graph with
  realistic benign sharing around them.
- A giveaway artifact exists → remove the artifact (round amounts, shared
  timestamps, sequential IDs, unique-device tells).

Never respond to a failed readiness check by changing the evaluation metric,
loosening a criterion, or proceeding anyway. The gate is the point.

Each check was confirmed to be capable of failing before its PASS was accepted.
Nine defects were seeded into copies of the data and the point-in-time module — a
duplicated row, a validation split stripped of fraud, fraud collapsed onto one
day, a warm-up shorter than the feature window, a feature module reading
`is_fraud`, a benign-sharing baseline erased, a ring cut loose from the graph,
money-flow links removed, and a history window whose upper bound reached to the
end of the dataset — and all nine were caught by the check that claims to catch
them. A gate that cannot go red is decoration.

---

## Related documents

- `docs/FEATURE_LEAKAGE_POLICY.md` — point-in-time and quarantine rules
- `docs/DATA_SPLIT_PROTOCOL.md` — split construction and sealing
- `DATA_DICTIONARY.md` — entity and transaction schema, ground truth
- `docs/MODEL_BASELINES.md` — where naive rules reappear as reference points
- `ROADMAP.md` — Checkpoint 1.5 placement
- `PRD.md` — §4 detection classes, §11 limitations
