# Paper 2 implementation audit for Paper 3 (source-grounded)
Reference: `paper2/experiment.py`, `paper2/dev_loso.py`, `paper2/validate_utd_archives.py`, `paper2/real_data_smoke.py`, `paper2/model_smoke.py`, `paper2/protocol.yaml` at commit `ab2a0319220ce114ba22f340a9eaae33ed6915eb`. This document describes inspected source, not execution results.

## Dataset is fixed: UTD-MHAD, *not* NTU RGB+D
- Original paired modalities: official UTD-MHAD RGB videos (`RGB.zip`) and 3D skeleton MATLAB files (`Skeleton.zip`), matched via regex `a(\d+)_s(\d+)_t(\d+)` on archive member basename.
- 27 action IDs (mapped to labels 0..26), 8 subjects, expected 861 synchronized trials. Archive validator checks 861 paired identifiers and zero unmatched RGB/skeleton identifiers.
- Fixed subject-independent split: subjects 1,3,5,7 (431 samples) for development; subjects 2,4,6,8 (430) held out for final evaluation. LOSO folds across development subjects only, seeds 42,123,2026.
- `decode_rgb`: read complete video with OpenCV, uniformly sample 8 frames, resize to 112 x 112, normalize by hard-coded means [0.43216,0.394666,0.37645] and stds [0.22803,0.22145,0.216989]. Expected tensor `[B,3,8,112,112]`.
- `decode_skel`: load MATLAB variable `d_skel`, transpose to time/joint/coordinate, uniformly sample 32 steps, subtract joint 0 root per step, normalize by average joint distance. Expected tensor `[B,32,20,3]`.
- This exact preprocessing is the first reproducible Paper 3 baseline; do not silently switch to 64x64 unnormalized smoke-test inputs. **Audit source of RGB normalization constants** to confirm no held-out data contributed to statistics.

## Actual Paper 2 production architecture
- `RGBEncoder`: Conv3d(3,32) -> BatchNorm3d/ReLU/MaxPool3d -> Conv3d(32,64)/ReLU -> global adaptive pooling -> linear(64,128).
- `SkeletonEncoder`: flattened 20x3 joint channels -> Conv1d(60,128) -> ReLU -> Conv1d(128,128) -> ReLU -> global temporal mean.
- Skeleton-motion stream: first difference of 32-step skeleton with repeated first step, absolute time-mean (60 values), linear(60,128)/ReLU. This is a **derived skeleton feature, not a physically independent third sensor**.
- `FaithfulHAR`: 3-stream learned softmax gate conditioned on the concatenation of RGB/skeleton/motion feature vectors; weighted sum -> linear 27-class logits; development training applies 0.2 stochastic feature-stream dropout.
- `modality_faithfulness` evaluates true-class confidence differences under `force_drop` feature zeroing. It does not measure calibrated reliability, and the gate is still computed from unmasked original features during `force_drop` inference.
- `dev_loso.py`: cache all 431 development subjects only, train 12 subject/seed folds for 20 epochs by default (batch 16, AdamW lr 3e-4), select best macro-F1 epoch; log fold metrics and aggregate summary. It **does not save model checkpoint weights**.
- `model_smoke.py` and `real_data_smoke.py` use a smaller 64-dimensional smoke model, not the production `FaithfulHAR`; do not treat their weights or architecture as the production baseline.

## Specific Paper 3 engineering implications
1. Start from `FaithfulHAR` and the production `UTD` loader, not the standalone minimal smoke model. Preserve exact dimensions and identifiers; add new Paper 3 files rather than editing original source.
2. Introduce explicit RGB-present and skeleton-present binary masks. If skeleton is absent, **mask both skeleton and motion streams**. Do not treat derived motion as an independent functioning sensor.
3. Recompute reliability/gating after corruption or missingness is applied. An unavailable stream must receive zero fusion weight and remaining active stream weights must renormalize.
4. Train a reliability estimator on development-only corruptions with labels unavailable at deployment. Test calibration and confidence under corrupted inputs. Compare against fixed fusion, Paper 2 unmodified, attention gate and modality-dropout baseline.
5. Predefine corruption operators and severity for RGB video and skeleton time-series; apply physically meaningful perturbations without leaking test data. Report clean/isolated/dual failure performance, macro-F1, NLL, ECE, Brier, selective risk-coverage and runtime.
6. The existing `force_drop` changes features after encoder execution and so **does not save inference compute**. Measure true runtime for absent-sensor handling, not estimated FLOPs.
7. Confirm availability and provenance of any Paper 2 checkpoints. Do not claim a frozen trained model exists solely because the source code exists. Without weights, reproduce a trained reference from the pinned source under a predeclared protocol.
8. Strictly preserve original UTD-MHAD data licensing and avoid committing data archives into GitHub.

## Verification required before GPU
- Match 861 archive keys with no duplicates or unmatched members and confirm 431/430 split.
- Unit tests for RGB-only, skeleton-only, both missing (abstain or explicit failure), both present, and combinations of corruption.
- Ensure zero-weight masking is numerically stable and no test-domain statistics leak into normalization/calibration.
- Validate per-subject LOSO and clean-vs-corrupt comparisons; log seed/commit/dataset hashes and archive checkpoints.
- Do not dispatch Actions, download data, or provision GPU merely to add this design document.
