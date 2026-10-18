# Paper 3 methodology draft (not frozen)
## Target task and preregistration hypothesis
Given paired RGB sequence `R` and skeleton sequence `S`, predict human action `y` while handling degraded or absent sensors. Primary hypothesis: reliability-aware fusion improves robustness and calibration under unseen faults compared with a frozen Paper 2 model and matched alternative fusion baselines without unacceptable clean-data degradation.
## Dataset and split
Initial development dataset: UTD-MHAD (27 actions, 861 paired samples), matching the **frozen** Paper 2 subject partition: development subjects 1,3,5,7 (431), final test subjects 2,4,6,8 (430). Perform LOSO model selection *only* inside development subjects. Seeds: 42,123,2026. Do not query final-test labels to select corruptions, severity, fusion design, calibration, or hyperparameters. If external validation is added, define it before evaluation, handle label-space mismatch explicitly, and check data licensing.
## Fault taxonomy (to validate for realism)
- RGB: blur, low light, compression artifacts, partial occlusion, random frame drop, sustained missing intervals.
- Skeleton: missing joints, contiguous joint dropout, coordinate noise, temporal jitter, swapped or implausible joints if physically appropriate.
- Dual-modal: correlated faults, asynchronous failures, one sensor fully missing.
Define severity with signal-level quantities and label-preserving checks; do not assume arbitrary perturbations are physically realistic.
## Method candidates
- Frozen or fine-tuned Paper 2-compatible backbone as appropriate; compare both fairly.
- Per-modality observation quality and predictive uncertainty (not simply attention weight).
- Reliability-aware gating, optional abstention, and consistency constraints under perturbations.
- Compare entropy/margin, modality dropout training, static fusion, uncertainty-gated fusion, and a compute-matched fusion baseline.
Formal loss, epistemic/aleatoric separation, temperature-scaling protocol and intervention validity must be specified **before training**.
## Evaluation
Primary: mean macro-F1 across preregistered corruption families/severities (with clean macro-F1 guardrail).
Secondary: degradation slopes, worst-group macro-F1, ECE (fixed bins), NLL, Brier score, risk-coverage/AURC, missing-modality performance, latency and memory. Subject-level paired uncertainty intervals where statistically defensible; avoid treating correlated clips as independent.
Analysis: per-class and per-subject tables; confidence under failure; error taxonomy; corrupted vs uncorrupted controls; calibrated reliability versus raw attention weights.
## Baselines and ablations
- Frozen Paper 2 reference at pinned SHA (only if runnable).
- RGB-only, skeleton-only, late fusion, attention fusion, modality-dropout training, reliability oracle (upper bound only).
- Remove reliability head, remove fault-aware training, remove cross-modal constraint, vary severity and sensor availability.
## Freeze criteria
1. Literature comparison matrix with precise novel claim and overlaps.
2. Fixed corruption recipes and development-only validation plan.
3. Frozen metrics, primary endpoint, error bars and computing budget.
4. Reproducible configs, pinned dependencies, seed logging and durable artifact archival.
5. Explicit approval before workflow dispatch or paid GPU training.
