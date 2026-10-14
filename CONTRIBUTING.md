# Contributing to Abuse-Ring Sentinel

This is a research/portfolio prototype, but it is developed with production-style
discipline. The rules below exist because fraud-detection code is unusually easy
to get *subtly* wrong — a single leaked future value can turn a mediocre model
into one that looks excellent and is worthless.

## Getting set up

```bash
uv sync --extra all      # venv + all dependencies
cp .env.example .env     # then edit as needed
uv run pytest            # should pass before you start
```

## Non-negotiable engineering rules

These are enforced by tests and reviewed on every change.

**1. Synthetic data only.** No real financial data, no real customer data, no real
PII — not in code, not in tests, not in fixtures, not in documentation examples.

**2. No fabricated metrics.** If a number was not produced by an executed
experiment, it does not go in a document, a docstring, a commit message, or a
README. Write `Not yet measured.` instead. This applies to precision, recall,
PR-AUC, cost, latency, detection rates, and speedups.

**3. Point-in-time safety.** For a transaction at time `T`, only data with
timestamp `< T` may influence its features. Windows are always half-open and
backward-looking: `[T - Δ, T)`. Any new feature must ship with a leakage test.

**4. The held-out test set is sealed.** It is not used for training, feature
fitting, scaling, threshold selection, hyperparameter search, early stopping, or
"just checking". It is opened once, at Checkpoint 6. See
`docs/DATA_SPLIT_PROTOCOL.md`.

**5. Preprocessing is fit on train only.** `fit(X_train)` then `transform(...)`
everything else. Never `fit` on validation or test.

**6. No hardcoded fraud decisions.** No `if amount > 10000: fraud`. Detection
comes from models, configured rules with documented thresholds, or graph signals.

**7. No secrets in source.** Configuration comes from environment variables.
`.env` is gitignored; `.env.example` holds placeholders only.

**8. Evidence must be real.** Every item in an explanation must map to an actual
feature value, model attribution, or graph finding. Never generate plausible-
sounding evidence.

**9. Humans decide consequential actions.** The system may score, alert, and
recommend. It must not autonomously block a customer.

## Checkpoint workflow

Work proceeds in explicit checkpoints (see `ROADMAP.md`). For each one:

1. Implement
2. Test
3. Verify
4. Document
5. Stop and report

Do not begin the next checkpoint until the current one is verified and reported.
Partial work across three checkpoints is worse than one finished checkpoint.

## Before you open a pull request

```bash
make check      # ruff + mypy + pytest
```

`uv run pytest` must pass. Leakage tests (`-m leakage`) must never be skipped,
xfailed, or deleted to make a build green — if a leakage test fails, the feature
is wrong, not the test.

## Commit messages

Use a short imperative subject with a scope prefix matching the component:

```
simulator: add configurable cycle length to circular money flow
ml: fit velocity features on half-open backward windows
graph: detect shared-device communities via Louvain
docs: record Checkpoint 2 validation results
```

State what changed and why. If a commit records experimental results, say which
split they came from.

## Adding a dependency

Justify it. This project deliberately prefers a smaller dependency set over
convenience:

- Do not add Neo4j unless NetworkX has been shown insufficient.
- Do not add graph neural networks unless the simpler graph approach has been
  implemented, evaluated, and measured as inadequate.
- Do not add a deep-learning framework for tabular baselines.
- Community detection uses `networkx.algorithms.community.louvain_communities`,
  which is built in — no extra package required.

## Documentation expectations

Code changes that alter behaviour update the relevant doc in the same change.
Features update `docs/FEATURE_REFERENCE.md` with full provenance metadata
(source, window, aggregation, point-in-time safety). Model changes update
`MODEL_CARD.md`. Limitations are documented honestly and prominently.
