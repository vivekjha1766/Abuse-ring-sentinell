# Project Status

**As of:** 2026-09-06

This is the authoritative status summary. Individual checkpoint documents retain
their historical status and verification narrative; when a historical header and
this document differ, this document describes the current repository state.

## Current State

| Area | State | Evidence |
|---|---|---|
| Synthetic simulator | Implemented and frozen for the benchmark | `simulator/`, `docs/CHECKPOINT_1_VERIFICATION.md` |
| PIT features and temporal engine | Implemented and leakage-tested | `ml/`, `tests/leakage/` |
| Preprocessing and baseline models | Implemented; new runs capture provenance | `ml/`, `scripts/train_baseline_models.py` |
| Checkpoint 2H transaction benchmark | Frozen and sealed; reported once | `README.md`, `docs/SUBMISSION_RESULTS.md` |
| Graph layer | Investigation/demo graph only; not fused into scoring | `graph/`, `docs/SUBMISSION_RESULTS.md` |
| Backend | Read-only demo API; no persistence or authentication | `backend/` |
| Frontend | Static demo; React application not implemented | `frontend/demo/`, `frontend/src/` |
| Entity reputation and risk fusion | Not implemented | Target architecture only |

## Evidence Boundaries

- Checkpoint 2H heldout data and result payloads are sealed historical evidence.
- The frozen metrics must not be recomputed, tuned, or changed.
- New experiments may use only trainable non-heldout inputs and must write a
  provenance manifest containing dataset, configuration, schema, split, runtime,
  and threshold-selection metadata.
- The frozen benchmark numbers remain exactly as reported in `README.md` and
  `docs/SUBMISSION_RESULTS.md`.

## Planned Work

Graph scoring and PIT-safe graph features, reputation, fusion, persistence,
security hardening, monitoring, CI/deployment, and the planned React frontend
remain future work. Their specifications are retained in `ARCHITECTURE.md`,
`ROADMAP.md`, and `PRD.md`; they are not claims about the current implementation.
