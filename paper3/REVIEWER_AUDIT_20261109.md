# Paper 3 reviewer audit — 2026-11-09
Branch: `paper3/reviewer-audit-20261109`. This branch does not alter Paper 2 or main.

## Inspected source
- `src/paper3/core.py`: temporal fusion and feature/pixel corruption operators.
- `src/paper3/reviewer_campaign.py`: three-seed Penn Action training, corruptions, predictions, paired inference ablations, exploratory statistics.
- `src/paper3/final_campaign.py`: independently trained mask-aware and no-quality-supervision controls, per-class and clean/corrupt reliability diagnostics.
- `PROTOCOL_PENNACTION_REVIEWER_CAMPAIGN.md`: controlling protocol. The top-level README and `PROTOCOL_DRAFT.md` contain older planning descriptions and must not override this protocol.

## Verified code-level findings (not GPU results)
1. V2 and specialist_rule are identical under explicit complete modality absence by construction. Their equal missing-modality scores are expected; no reliability-supervision gain can be inferred from them.
2. `specialist_quality_no_supervision` in reviewer_campaign is an inference-time sensitivity configuration, not independently retrained without supervision. `final_campaign.py` separately trains a no-quality-supervision control.
3. `reviewer_campaign.py` computes quality-detection AUROC from pooled corrupted categories without a clean negative class; it measures discrimination among corruption categories, not clean-versus-corrupted detection. Use the paired clean/corrupt diagnostics from final_campaign and label their synthetic scope.
4. Feature-to-logit latency omits ResNet-18 image extraction, data loading, and pose generation. Do not claim end-to-end efficiency.
5. Previous official Penn Action test inspection informed development. All further Penn Action test-set comparisons remain exploratory, not independent confirmation.
6. `core.py` contains an older `FailureResilientFusion` representation-residual architecture; do not confuse its results with the separately implemented V2 router or mask-aware logit adapter.

## Implemented in this branch
`src/paper3/artifact_audit.py` validates all exported per-video predictions against fixed test cardinality, aligned sample identities and labels, correct correctness flags, valid class indices, and valid confidence. `src/tests/test_artifact_audit.py` adds five unit tests. No GPU experiment has been run as part of this audit.

## Execution gate
Run `cd paper3/src && python -m unittest discover -s tests -v`.
After generating `predictions_by_video.csv`, run:
`python -m paper3.artifact_audit --predictions OUTPUT/predictions_by_video.csv --out OUTPUT/artifact_audit.json`
Repeat for combined final prediction outputs if exported. Archive exact source SHA, manifest hash, checkpoints, per-video predictions, logs, and all diagnostics before deleting GPU resources.

## Still required for stronger journal evidence
- Independent held-out evaluation not used in method selection, preferably a compatible second dataset and measured predicted-pose errors.
- Explicit quality-label polarity and calibration verification.
- Fully specified published baselines under matched conditions.
- End-to-end latency and memory profiling, including the image feature extractor.
- Full paired statistical reporting, with dependence across seeds acknowledged.
- No fabricated or inferred experimental measurements.
