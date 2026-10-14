# Feature Reference

**Status:** Checkpoint 0 — specification. Features are implemented in Checkpoint 2.
**Owner:** ML Agent (see `AGENTS.md`)
**Implementation:** `ml/features.py`, registry in `ml/feature_registry.py`,
temporal signals in `ml/temporal.py`

**Purpose.** The authoritative catalogue of every feature, with full provenance.
A feature that is not documented here does not exist as far as the project is
concerned.

---

## 1. The provenance schema

Every feature carries five mandatory metadata fields, declared in the feature
registry:

| Field | Meaning |
|---|---|
| `feature_name` | Stable identifier used in code, artifacts, and reports |
| `source` | The data the feature reads (e.g. `account_transactions`) |
| `window` | Historical window width, or `none` for point features |
| `aggregation` | How the window is reduced (`count`, `sum`, `zscore`, `none`, …) |
| `point_in_time_safe` | Whether the feature respects the strict `ts < T` rule |

Canonical example from the project specification:

```yaml
tx_velocity_1h:
  source: account_transactions
  window: 1h
  aggregation: count
  point_in_time_safe: true
```

This metadata is not decoration. It makes leakage auditable (every feature must
declare `point_in_time_safe: true` and prove it with a test), makes the ablation
study mechanical (feature groups are selected by metadata, not by hand-maintained
lists), and makes explanations traceable (an evidence item can name the window and
aggregation behind the number it cites).

---

## 2. Feature catalogue

All windows are **half-open and backward-looking**, `[T − Δ, T)`. All features
below are `point_in_time_safe: true`; any future feature that cannot be is
rejected rather than documented.

### 2.1 TRANSACTION group

Properties of the transaction itself. No history required.

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `transaction_amount` | float | `transaction` | none | none | n/a (always present) | The transaction's monetary amount |
| `log_transaction_amount` | float | `transaction` | none | none | n/a (always present) | `log1p(amount)`; compresses the heavy right tail |

### 2.2 VELOCITY group — transaction counts

How *often* the account has transacted recently. Targets bursts and automation.

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `tx_velocity_15m` | int | `account_transactions` | 15m | count | `0` | Count in the prior 15 minutes |
| `tx_velocity_1h` | int | `account_transactions` | 1h | count | `0` | Count in the prior hour |
| `tx_velocity_6h` | int | `account_transactions` | 6h | count | `0` | Count in the prior 6 hours |
| `tx_velocity_24h` | int | `account_transactions` | 24h | count | `0` | Count in the prior 24 hours |
| `tx_velocity_7d` | int | `account_transactions` | 7d | count | `0` | Count in the prior 7 days |

### 2.3 VOLUME group — summed amounts

How *much* has moved recently. Distinguishes many-small from few-large activity.

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `tx_volume_15m` | float | `account_transactions` | 15m | sum | `0.0` | Summed amount, prior 15 minutes |
| `tx_volume_1h` | float | `account_transactions` | 1h | sum | `0.0` | Summed amount, prior hour |
| `tx_volume_6h` | float | `account_transactions` | 6h | sum | `0.0` | Summed amount, prior 6 hours |
| `tx_volume_24h` | float | `account_transactions` | 24h | sum | `0.0` | Summed amount, prior 24 hours |
| `tx_volume_7d` | float | `account_transactions` | 7d | sum | `0.0` | Summed amount, prior 7 days |

### 2.4 BEHAVIOR group — deviation from personal baseline

Relative, not absolute. This is what prevents "large amount ⇒ fraud".

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `amount_deviation` | float | `account_transactions` | historical (to `T`) | zscore | `0.0` | How far this amount sits from the account's own historical amount distribution |
| `tx_frequency_deviation` | float | `account_transactions` | historical (to `T`) | zscore | `0.0` | How far current transaction frequency sits from the account's historical frequency |

### 2.5 GEOGRAPHY group

Physical plausibility of consecutive activity.

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `geo_distance_km` | float | `account_transactions` | prior transaction | none | `0.0` | Great-circle distance from the account's previous transaction location |
| `geo_speed_kmh` | float | `account_transactions` | prior transaction | derived | `0.0` | `geo_distance_km ÷ elapsed_hours` |
| `impossible_travel` | int (0/1) | derived | prior transaction | threshold | `0` | `1` when `geo_speed_kmh` exceeds the configured feasible-speed limit |

### 2.6 ENTITY group

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `account_age_days` | float | `account` | none | none | `0.0` | Days between account creation and `T` |

### 2.7 TEMPORAL INTELLIGENCE group

Rolling z-scores from `ml/temporal.py`. Baseline mean and standard deviation are
computed from *historical* windows only.

| `feature_name` | dtype | `source` | `window` | `aggregation` | `cold_start_default` | Description |
|---|---|---|---|---|---|---|
| `temporal_zscore_count_15m` | float | `account_transactions` | 15m rolling | zscore | `0.0` | Burst score on counts, 15m |
| `temporal_zscore_count_1h` | float | `account_transactions` | 1h rolling | zscore | `0.0` | Burst score on counts, 1h |
| `temporal_zscore_count_6h` | float | `account_transactions` | 6h rolling | zscore | `0.0` | Burst score on counts, 6h |
| `temporal_zscore_volume_15m` | float | `account_transactions` | 15m rolling | zscore | `0.0` | Burst score on volume, 15m |
| `temporal_zscore_volume_1h` | float | `account_transactions` | 1h rolling | zscore | `0.0` | Burst score on volume, 1h |
| `temporal_zscore_volume_6h` | float | `account_transactions` | 6h rolling | zscore | `0.0` | Burst score on volume, 6h |

---

## 3. Detection intent per group

Which patterns each group is meant to catch, and which attack families it should
help with.

| Group | Targets | Attack families it should help detect |
|---|---|---|
| TRANSACTION | Raw magnitude, tail behaviour | Weak alone by design; supports others |
| VELOCITY | Automation, bursts, scripted activity | Transaction spike, coordinated microtransactions, account takeover |
| VOLUME | Value throughput independent of count | Transaction spike, account takeover |
| BEHAVIOR | Departure from the entity's *own* norm | Account takeover, unusual-amount anomalies |
| GEOGRAPHY | Physical impossibility | Account takeover, credential sharing |
| ENTITY | Immaturity as a risk factor | Synthetic identity cluster, fake merchant ring |
| TEMPORAL | Statistical bursts vs own history | Transaction spike, coordinated microtransactions |

Note the deliberate gap: **no feature in this catalogue can see a relationship
between two entities.** Velocity and volume are per-account; deviation is
per-account; geography is per-account. A fake merchant ring in which every
individual transaction is plausible produces no strong signal in any feature
above. That gap is the project's central hypothesis, and closing it is the job of
the graph layer in Checkpoint 3 — not of a cleverer transaction feature.

---

## 4. Cold-start policy

New entities have no history. Uncontrolled `NaN`/`inf` is forbidden — it silently
poisons downstream math and hides bugs. Every feature has a documented default,
and a `cold_start` indicator flag marks affected rows so both models and analysts
can see when a score rests on thin history.

| Situation | Affected features | Handling | Rationale |
|---|---|---|---|
| Account with zero prior transactions | all velocity, volume | `0` / `0.0` | Genuinely no prior activity; zero is the truthful value |
| Account with zero prior transactions | `amount_deviation`, `tx_frequency_deviation` | `0.0` | No baseline exists, so "no deviation" is the neutral claim |
| History shorter than window `Δ` | velocity, volume | Compute over available history (still `< T`) | Partial history is real information |
| Zero standard deviation in baseline | any `zscore` feature | `0.0`, never `x/0` | Constant history means no statistical surprise |
| Insufficient history for a z-score baseline | temporal z-scores | `0.0` + `cold_start = 1` | Avoid manufacturing extreme scores from 2 observations |
| No prior transaction location | `geo_distance_km`, `geo_speed_kmh` | `0.0` | Nothing to measure movement against |
| No prior transaction location | `impossible_travel` | `0` | Cannot assert impossibility without a reference point |
| Brand-new account | `account_age_days` | `0.0` | Truthful |

| Indicator | dtype | Meaning |
|---|---|---|
| `cold_start` | int (0/1) | `1` when the row's history falls below the documented minimum for reliable statistics |

**Invariant:** no feature ever emits `NaN` or `inf`. A test asserts finiteness
across the entire feature matrix.

---

## 5. Derived-feature rules and definitions

**Why `log1p` on amount.** Transaction amounts are heavy-tailed and
right-skewed. A linear model fed raw amounts is dominated by the tail; `log1p`
compresses it while handling zero safely. Both raw and log forms are kept so
tree models (which are scale-invariant) and linear models (which are not) each
have a usable representation.

**Why both count and volume windows.** They separate two different abuse shapes.
Coordinated microtransactions raise *count* while keeping *volume* modest.
A single large fraudulent transfer raises *volume* with no change in *count*.
Keeping both lets the model distinguish them; a combined "average amount" feature
would blur exactly the distinction that matters.

**Why five window scales.** Different attacks operate on different clocks. A
scripted burst shows at 15 minutes and may be invisible at 7 days. A slow
account-takeover ramp shows at 24 hours or 7 days and is invisible at 15 minutes.
The multi-scale set is a deliberate hedge against assuming an attack tempo.

**`impossible_travel` definition.** A boolean derived from `geo_speed_kmh`
exceeding a configured feasible-speed threshold. The threshold is a **named
configuration constant** (`IMPOSSIBLE_TRAVEL_MAX_KMH` in `simulator/config.py` /
feature config), not a magic number inline in the feature code, so it is visible,
documented, and tunable.

Honest caveats on this feature: legitimate causes of apparent impossible travel
include VPNs and proxies, corporate egress routing, shared accounts used by family
members in different places, card-not-present purchases from a merchant in another
country, and IP geolocation error. `impossible_travel = 1` is therefore *evidence*,
never proof, and it is presented to analysts as one signal among several.

**Z-score computation.** For a value `x` against historical baseline mean `μ` and
standard deviation `σ`, both computed from data with `ts < T`:

```
z = (x − μ) / σ        if σ > σ_min and n_history ≥ n_min
z = 0.0                otherwise
```

The guards matter. `σ = 0` (a perfectly regular account) would otherwise produce
a division by zero, and a tiny `n_history` would produce wild z-scores from two
data points. Both cases return `0.0` and set `cold_start`.

---

## 6. How to add a feature

1. Implement it in `ml/features.py` (or `ml/temporal.py`), taking an explicit
   `as_of` timestamp and reading only through the history view.
2. Register it in `ml/feature_registry.py` with all five provenance fields.
3. Define and document its cold-start default; guarantee no `NaN`/`inf`.
4. Add at least one leakage test to the `-m leakage` suite.
5. Add its row to the catalogue in this document, in the same change.
6. Confirm it does not import the ground-truth registry.

See the reviewer checklist in `docs/FEATURE_LEAKAGE_POLICY.md` §6.

---

## 7. Measured feature importance

Not yet measured. (Populated in Checkpoint 2 from model attributions on the
validation split, and revisited in Checkpoint 6.)

---

## Related documents

- `docs/FEATURE_LEAKAGE_POLICY.md` — point-in-time rules and required tests
- `docs/MODEL_BASELINES.md` — which models consume which feature groups
- `docs/EXPERIMENT_PROTOCOL.md` — ablation variants built from feature groups
- `docs/ML_READINESS.md` — pre-modelling data audit
- `DATA_DICTIONARY.md` — underlying entity and transaction fields
- `PRD.md` — FR-4 point-in-time feature engine, FR-5 temporal intelligence
