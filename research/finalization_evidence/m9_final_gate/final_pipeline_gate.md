# Final Pipeline Gate Evidence

Created at UTC: `2026-09-22T02:57:48.579505+00:00`

Decision: `PASS`

Handoff: `FINAL PIPELINE READY -> APPLICATION`

## Frozen runtime identity

- Model ID: `RF-REF-100-GINI-SQRT-UNPRUNED-CW`
- Model SHA-256: `61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f`
- Preprocessing SHA-256: `c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98`
- Encoded feature width: `47`
- Threshold: `0.50`
- Comparator: strict `>`

## Verification

- final pipeline has no runtime import from research: PASS
- official artifact fingerprints: PASS
- canonical 47-feature preprocessing contract: PASS
- dataset replay fast boundary: PASS
- interactive/cold-start inference: PASS
- deterministic repeated inference: PASS
- five clean walkthroughs: PASS
- full unit/regression suite: 175 tests, PASS
- training/rebuild does not overwrite official artifacts: PASS
- official artifact fingerprints unchanged after verification: PASS
- application clean-surface boundary: PASS

## Product surfaces

`final_pipeline/` contains the canonical executable ML system.

`application/` contains only the user-facing product surface.

Milestone/progress/gate evidence is intentionally retained under `research/finalization_evidence/`.
