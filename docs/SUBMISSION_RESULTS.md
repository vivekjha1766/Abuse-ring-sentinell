# Submission Results

All figures in this document are copied verbatim from the Checkpoint 2H sealed-
heldout evaluation (`scripts/evaluate_heldout.py`), run exactly once after every
model, hyperparameter, and decision threshold had already been frozen on the
validation split. Nothing here is recomputed. All data is synthetic.

## 1. Problem

Fraud detection systems that score one transaction at a time miss **coordinated
abuse rings**: groups of accounts that share devices, IPs, or merchants to move
money in ways that look unremarkable transaction-by-transaction. Abuse-Ring
Sentinel targets both individual anomalies (via ML) and coordinated rings (via
graph analysis of entity relationships).

## 2. Why conventional per-transaction detection misses coordinated fraud

A per-transaction feature vector has no place to represent "this device is also
used by nineteen other accounts that all pay the same four merchants." Each
individual transaction in a ring can have a plausible amount, hour, and merchant.
The evidence for the ring exists only in the relationships between transactions,
not inside any single one — which is why this project adds a graph investigation
layer alongside the transaction-level model rather than relying on either alone.

## 3. Solution architecture

```
simulator/  -> ml/ (PIT-safe features, temporal, preprocessing, models, eval)
            -> graph/ (investigation graph, ring candidates)
ml/ + graph/ -> backend/ (FastAPI demo API, frozen XGBoost inference)
backend/    -> frontend/demo/ (static dashboard: risk explorer, ring explorer,
               model performance, guided investigation walkthrough)
```

The ML core (simulator, features, split policy, preprocessing, baseline models,
and the Checkpoint 2H evaluation) is frozen for this submission. The backend,
graph layer, and dashboard are read-only consumers of that frozen core: they
never retrain, never re-threshold, and never read the sealed heldout split.

## 4. ML methodology

- **Point-in-time safety.** Every feature for a transaction at time `T` reads
  only data with `ts < T` (half-open backward windows). Enforced and leakage-
  tested; see `docs/FEATURE_LEAKAGE_POLICY.md`.
- **Strict, sealed splits.** Train (85,985 rows) and validation (28,434 rows)
  were used for all iteration; heldout (28,069 rows) was opened exactly once,
  at Checkpoint 2H, after every threshold was already frozen. See
  `docs/DATA_SPLIT_PROTOCOL.md`.
- **Cost-sensitive thresholding.** Each model's decision threshold was selected
  on validation by minimizing `FP x FP_COST + FN x FN_COST` (`FP_COST=15`,
  `FN_COST=450`), not by accuracy or a fixed 0.5 cutoff. See
  `docs/COST_MODEL.md`.
- **Three frozen baselines.** Majority-class, Logistic Regression (scaled,
  fit on train only), and XGBoost (fixed conservative hyperparameters, no
  tuning, `scale_pos_weight` computed from train labels only, unscaled
  features). See `docs/MODEL_BASELINES.md`.

## 5. Graph / ring investigation layer

`graph/build.py` constructs a heterogeneous graph (customer/account/device/ip/
merchant nodes; OWNS/USES/LOGGED_IN_FROM/TRANSACTS_WITH edges) from non-heldout
transactions. `graph/analytics.py` reports shared-device/shared-ip account
counts and connected-component "ring candidates" with a fraud-concentration
signal computed from non-heldout labels, for investigator context only. This
layer never feeds back into, or alters, the frozen XGBoost score or threshold —
it is a separate, investigation-only signal surfaced next to the model's score
in the dashboard.

## 6. Final results (frozen, Checkpoint 2H sealed heldout, evaluated once)

| Model | PR-AUC (primary) | ROC-AUC | Precision | Recall | F1 | Benchmark cost |
|---|---|---|---|---|---|---|
| Majority baseline | 0.0508 | 0.5000 | 0.0508 | 1.0000 | 0.0966 | $399,660 |
| Logistic Regression | 0.5864 | 0.9023 | 0.1547 | 0.8810 | 0.2623 | $179,490 |
| **XGBoost (primary model)** | **0.7430** | **0.9603** | **0.2460** | **0.9270** | **0.3888** | **$107,535** |

XGBoost is the primary detection model: highest PR-AUC and ROC-AUC, highest
recall among non-trivial models, and the lowest benchmark cost.

## 7. Business impact (framed only from the numbers above)

Under the synthetic `FP_COST=15` / `FN_COST=450` assumptions, the XGBoost model's
benchmark cost ($107,535) is 73% lower than the majority baseline ($399,660) and
40% lower than logistic regression ($179,490) on the same heldout set, while
catching 92.7% of fraud (recall) at 24.6% precision. These are relative
comparisons under one fixed, documented, synthetic cost assumption — not a claim
about real-world monetary savings.

## 8. Demo workflow

1. Start the backend (`uv run uvicorn backend.app.main:app --reload --port 8000`
   with `SENTINEL_EXPERIMENT_ID` set) and open `http://localhost:8000/demo/`.
2. **Executive Dashboard** — dataset size and the frozen heldout metrics above.
3. **Transaction Risk Explorer** — pick a transaction, see its fraud probability,
   risk band, decision, and the top contributing risk signals.
4. **Fraud Ring Explorer** — view the transaction's connected entities and any
   ring candidates sharing its device/IP.
5. **Model Performance** — Majority vs Logistic Regression vs XGBoost, side by
   side, over the same frozen numbers.
6. **Demo Investigation** — a guided walkthrough chaining steps 3-4 for one
   sample transaction, ending in a human-review recommendation (the system
   never blocks an account autonomously).

## 9. Limitations

- All data is synthetic; no real customer, account, or transaction exists in
  this pipeline.
- 5% prevalence and the FP/FN cost values are benchmark choices, not real-world
  or industry figures; absolute costs have no external meaning, only relative
  comparisons under identical assumptions.
- The graph/ring layer is investigative and demo-only; it is not fused into the
  model's score or threshold in this submission.
- The dashboard is a dependency-free static page built for this submission
  window, not the originally planned React/TypeScript frontend (still an empty
  scaffold under `frontend/src/`).
- No adversarial adaptation: simulated attackers do not react to being detected.
- This is a portfolio prototype, not a system validated for real financial
  decisioning.

## 10. Future work

- Fuse the graph ring signal into a unified risk score (with its own leakage
  and PIT safety review), rather than presenting it side by side with the ML
  score as in this submission.
- Replace the static dashboard with the planned React/TypeScript frontend.
- Extend the ablation study (transaction -> +temporal -> +graph -> +reputation)
  to measure each layer's marginal contribution on Class A vs Class B fraud,
  as described in `PRD.md`.
