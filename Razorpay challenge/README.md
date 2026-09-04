# Abuse-Ring Sentinel

**Graph-Based Fraud Intelligence & Coordinated Abuse Detection Platform**

A production-style portfolio prototype that detects both individual transaction
anomalies **and** coordinated fraud rings, on fully synthetic data.

> **Status: Checkpoint 1.5 of 8 complete — synthetic data, attacks, ground truth,
> and the ML-readiness gate.**
> The simulator generates a reproducible labelled dataset and passes an
> adversarial shortcut audit with zero oracle findings
> (`docs/CHECKPOINT_1_VERIFICATION.md`). The readiness gate then passes at 1%, 5%,
> 10%, and 25% prevalence, and the point-in-time, split, and evaluation contracts
> features will be built against are in place and tested
> (`docs/ML_READINESS.md`). No features, models, or detection thresholds exist
> yet, so every *detection* metric in this repository reads
> `Not yet measured.` and will stay that way until the corresponding experiment
> actually runs.

---

## The problem

Most fraud-detection projects score one transaction at a time. That works for a
stolen card making unusual purchases — an individual anomaly, visible in a single
row of data.

It fails for organised abuse. When twenty accounts share three devices to funnel
money through four fake merchants, **every individual transaction can look
perfectly normal.** Reasonable amount, plausible merchant, ordinary hour, sensible
location. A per-transaction model sees twenty unremarkable customers, because the
feature vector for one transaction has no place to represent *"this device is also
used by nineteen other accounts that all pay the same four merchants."*

The fraud does not live in any transaction. It lives in the **relationships
between** them.

## The thesis

> Transaction-level ML detects individual anomalies. Graph intelligence detects
> coordinated abuse. A credible fraud platform needs both — and should be able to
> *measure* which one is doing the work.

This project is built to test that claim rather than assert it. Two detection
classes are defined and evaluated separately:

| | Class A — individual anomalies | Class B — coordinated abuse |
|---|---|---|
| **Example** | account takeover, sudden spike | fake merchant ring, shared-device ring, circular money flow |
| **Visible in one transaction?** | often yes | frequently no |
| **Expected detector** | transaction ML + temporal | graph intelligence |

An ablation study (transaction → +temporal → +graph → +reputation) measures each
layer's marginal contribution to each class. If the graph layer does not help on
Class B, the results will say so.

---

## Capability layers

```mermaid
flowchart LR
    L1["Transaction ML<br/><i>individual anomalies</i>"] --> L2["Temporal<br/><i>bursts & velocity</i>"]
    L2 --> L3["Graph<br/><i>coordination</i>"]
    L3 --> L4["Risk Fusion<br/><i>unified score</i>"]
    L4 --> L5["Investigation<br/><i>humans decide</i>"]
```

Four signals — transaction, temporal, graph, and entity reputation — are computed
independently and combined into a single 0–100 score with attached, countable
evidence. Component scores are kept separate so the ablation is possible and so
every alert can explain itself.

---

## Engineering commitments

These are the parts worth reviewing, and they are enforced by tests rather than by
good intentions:

**Point-in-time feature safety.** Every feature for a transaction at time `T` reads
only data with `ts < T` — half-open backward windows `[T−Δ, T)`. This includes the
graph: the scoring graph is rebuilt **as-of `T`**, because a graph built from the
full dataset would let a ring's future edges inform its own detection. A dedicated
leakage test suite exists that may never be skipped, xfailed, or deleted.
See `docs/FEATURE_LEAKAGE_POLICY.md`.

**A sealed held-out set.** Splits are strictly temporal (60/20/20) and the held-out
set is opened exactly once, at Checkpoint 6, with all models, hyperparameters, and
thresholds already frozen. All iteration happens on validation.
See `docs/DATA_SPLIT_PROTOCOL.md`.

**Cost-sensitive evaluation, not accuracy.** At 5% prevalence, predicting "never
fraud" scores 95% accuracy with zero recall. PR-AUC is the primary ranking metric;
thresholds are chosen by minimising `FP×FP_COST + FN×FN_COST` on validation, never
assumed to be 0.5. See `docs/COST_MODEL.md` and `EVALUATION.md`.

**An ML-readiness gate before any modelling.** Checkpoint 1.5 audits whether the
synthetic data is even worth modelling: naive rules like "large amount ⇒ fraud"
must be demonstrably *weak*, legitimate outliers and legitimate device/IP sharing
must exist, and no giveaway artifact may separate the classes. If the audit fails,
the fix is the generator — never the metric. The gate passes at all four
prevalences, and each of its checks was verified capable of failing by seeding the
defect it claims to catch. See `docs/ML_READINESS.md`.

**No fabricated numbers.** Any quantity that has not been measured reads
"Not yet measured." Every number that does appear carries its split, seed,
threshold, cost configuration, and the synthetic-data caveat.
See `docs/EXPERIMENT_PROTOCOL.md`.

**Humans decide.** The system's most severe autonomous action is raising a CRITICAL
alert and *recommending* a hold. It never blocks, holds, or closes an account on
its own. The LLM Copilot (added last, deliberately) summarises existing evidence
and is constrained from determining fraud, recommending blocks, or inventing facts.

---

## Planned architecture

```mermaid
flowchart TD
    SIM["simulator/<br/>synthetic ecosystem<br/>+ 7 attack families"] --> ML["ml/<br/>features · temporal<br/>LogReg · XGBoost"]
    SIM --> GR["graph/<br/>NetworkX · communities<br/>cycles · reputation"]
    ML --> FUS["risk fusion<br/>0–100 + evidence"]
    GR --> FUS
    FUS --> API["backend/<br/>FastAPI · PostgreSQL"]
    API --> UI["frontend/<br/>React · Cytoscape"]
    SIM -.ground truth<br/>evaluation only.-> EV["evaluation<br/>4 levels"]
```

**Stack.** Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic, PostgreSQL
(SQLite for local dev), NumPy/Pandas/scikit-learn/XGBoost, NetworkX,
React + TypeScript + Vite + Tailwind + Recharts + Cytoscape.js, Docker Compose,
`uv` for Python packaging.

**Deliberately absent:** Neo4j (NetworkX is sufficient at this scale), graph neural
networks, and deep learning. Each is gated behind a written demonstration that
simpler methods are insufficient. Full rationale in `ARCHITECTURE.md`.

---

## Repository layout

```
simulator/     synthetic ecosystem + 7 parameterized attack families   (Checkpoint 1)
ml/            safety contracts now; features and models next          (Checkpoints 1.5, 2)
graph/         graph construction, metrics, reputation, fusion          (Checkpoint 3)
backend/       FastAPI app, SQLAlchemy models, Alembic migrations       (Checkpoint 4)
frontend/      React + TypeScript dashboard                            (Checkpoint 5)
tests/         unit · ml · graph · api · leakage · integration · e2e
docs/          cost model, split protocol, leakage policy, protocols
scripts/       dataset generation and utilities
data/          generated datasets and splits (git-ignored)
artifacts/     models, preprocessing, reports (git-ignored)
```

Generated data and model artifacts are never committed — they are reproduced from a
seed and config.

---

## Documentation

Start here depending on what you want to know:

| Question | Document |
|---|---|
| What is being built and why? | `PRD.md` |
| How is it designed? | `ARCHITECTURE.md` |
| What happens next, and what's the plan? | `ROADMAP.md` |
| What does the data look like? | `DATA_DICTIONARY.md` |
| How is success measured? | `EVALUATION.md` |
| What are the models and their limits? | `MODEL_CARD.md` |
| Who owns which part of the build? | `AGENTS.md` |
| What rules does contributed code follow? | `CONTRIBUTING.md` |

Deeper specifications live in `docs/`: `COST_MODEL.md`, `ML_READINESS.md`,
`DATA_SPLIT_PROTOCOL.md`, `FEATURE_LEAKAGE_POLICY.md`, `FEATURE_REFERENCE.md`,
`EXPERIMENT_PROTOCOL.md`, `MODEL_BASELINES.md`.

---

## Getting started

```bash
cp .env.example .env      # non-secret defaults; set your own SECRET_KEY
uv sync --all-extras      # or: make install-all
uv run pytest             # simulator regression + leakage suites
make data                 # generate the dataset into data/ (seed 42, prevalence 5%)
python scripts/audit_ml_readiness.py --prevalence 0.05    # the Checkpoint 1.5 gate
```

`make help` lists the available targets. `make data` writes `data/raw/`
(transactions, entity tables, ground-truth registry, manifest, validation report)
and `data/splits/{warmup,train,validation,heldout}/`; the same seed and config
reproduce it byte-for-byte. The API and the dashboard become available at
Checkpoints 4 and 5.

---

## Results

**Not yet measured.**

No features have been engineered, no model has been trained, and no experiment has
been run. What *has* been established is that the data is fit to model and that the
safety machinery works: `docs/CHECKPOINT_1_VERIFICATION.md` records the dataset's
properties and the shortcut audit behind it, and `docs/ML_READINESS.md` records the
readiness gate passing at 1/5/10/25% with zero oracle findings and zero
point-in-time contaminations. Both are properties of the data and the scaffolding,
not evidence about a detector. The results tables in `EVALUATION.md`,
`docs/MODEL_BASELINES.md`, and `MODEL_CARD.md` are laid out and waiting, and every
cell says so honestly.

---

## Limitations

Stated up front rather than buried, because a reviewer will find them anyway:

- **All data is synthetic.** There is no real PII, no real transaction, and no real
  customer anywhere in the pipeline. The system can only detect abuse the generator
  produces; the generator's assumptions become the system's apparent strengths and
  its blind spots.
- **Circular validity.** The same project defines both the attacks and the
  detectors. Varying attack parameters across splits mitigates this but does not
  eliminate it.
- **5% prevalence is a benchmark choice**, not a real-world or industry fraud rate.
  Cost values (`FP_COST=15`, `FN_COST=450`) are likewise benchmark assumptions;
  absolute cost totals have no external meaning, only relative comparisons under
  identical assumptions.
- **Results are not externally comparable** to public fraud benchmarks or
  production systems.
- **No adversarial adaptation.** Simulated attackers do not react to being detected.
- **Prototype scale.** This is a **production-style portfolio prototype**, not a
  system validated for real financial decisioning in a banking environment.

---

## License

MIT — see `LICENSE`.
