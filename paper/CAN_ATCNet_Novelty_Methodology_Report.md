# CAN-ATCNet Novelty and Methodology Report

## Status
Working paper report based on the implemented notebook, completed 3-seed experiments, diagnostic run, and selective-combiner confirmation run through 7 October 2026.

## Central research question
Can predictive inconsistency in CAN payload/timing dynamics provide complementary security evidence to a supervised temporal attack classifier, and can that evidence improve robust multiclass CAN intrusion decisions without relying on indiscriminate score fusion?

## Implemented architecture
CAN-ATCNet uses a causal temporal encoder over CAN ID, payload, DLC/timing-derived features. The implemented system contains:

1. **Multiclass discriminative path**: predicts Normal, DoS, Fuzzy, Gear and RPM classes.
2. **Predictive path**: predicts next-step payload bytes and timing-related values from temporal state.
3. **Residual-derived consistency head**: prediction-error features are detached before the consistency head, preventing the attack objective from teaching the predictor to manufacture large residuals. The consistency head produces an attack-evidence score.
4. **Decision integration**: the classifier probabilities and consistency evidence can be fused. Experiments show that global fixed fusion is inferior to selective use.

Important terminology: the score used in fusion/diagnostics is the **residual-derived consistency-head attack score**, not the raw Smooth-L1 prediction residual itself.

## Defensible novelty
The primary novelty should NOT be stated as merely “detect attacks and identify attack type.” Detection plus attack classification, multitask CAN IDS, dual-branch architectures, and anomaly-plus-classification pipelines already exist.

The stronger contribution is the **joint discriminative-predictive security formulation**:
- a supervised temporal classifier learns attack identity;
- an auxiliary predictive model learns expected CAN payload/timing dynamics;
- prediction mismatch is transformed into a residual-derived consistency signal;
- this signal is used as complementary attack evidence;
- selective integration is empirically more effective than indiscriminate fusion.

### Contribution claims suitable for a manuscript
1. A causal CAN IDS that jointly learns attack discrimination and next-message behavioral prediction.
2. A residual-derived consistency mechanism that converts predictive mismatch into auxiliary attack evidence while detaching the residual features from the attack-loss gradient.
3. An empirical demonstration that the auxiliary consistency signal contains complementary decision information: selective integration improves accuracy, balanced accuracy and macro-F1 over TCN-CE on a later confirmation block across all three seeds.
4. A negative/diagnostic result that global residual fusion can harm ranking/calibration, motivating selective rather than universal fusion.
5. A leakage-conscious chronological evaluation protocol with train/validation/held-out separation, fixed seeds and later confirmation analysis.

## Why the novelty is more defensible than a two-head claim
Recent CAN-IDS literature already includes multitask learning, dual temporal/spatial branches, multimodal anomaly detectors, two-stage anomaly/classification systems, and auxiliary classifiers. Therefore architecture-count novelty (“we have two heads”) is weak.

The differentiator is the **role and training semantics of the second mechanism**: it is not simply another classifier. It models expected CAN behavior and derives attack evidence from predictive inconsistency. The paper should emphasize complementary evidence and the controlled gradient path, then prove its contribution through ablations.

## Literature pressure points to discuss
- 2026 Sensors: “A Multitask Learning Approach for Intrusion Detection in Controller Area Networks” establishes that multitask CAN IDS is not itself new.
- 2026 Kuwait Journal of Science: “A dual-branch temporal-spatial deep learning architecture for efficient and accurate CAN bus intrusion detection” establishes that dual-branch temporal/spatial architectures are not itself new.
- 2024 Journal of Information Security and Applications: “CANival: A multimodal approach to intrusion detection on the vehicle CAN bus” combines time-interval and signal-based anomaly analysis.
- 2026 Electronics: “A Two-Stage Hybrid Intrusion Detection System for CAN Bus Based on Statistical Thresholds and Random Forest Classifiers” combines anomaly gating with attack characterization.
- Auxiliary-classifier CAN IDS work already targets known-attack fine-grained classification and unknown-attack detection.
- Therefore the manuscript must avoid claiming first-ever dual-head, multitask, anomaly-plus-classification, or attack-type identification unless an exhaustive systematic review proves it.

## Ablation logic
The paper should present the mechanism incrementally:

| Variant | Multiclass classification | Predictive learning | Residual-derived consistency | Selective integration |
|---|---:|---:|---:|---:|
| TCN-CE | Yes | No | No | No |
| TCN-Focal | Yes | No | No | No |
| TCN-Predictive | Yes | Yes | No | No |
| CAN-ATCNet / fixed fusion | Yes | Yes | Yes | No |
| Selective CAN combiner | Yes | Yes | Yes | Yes |

This table is the core defense against “architectural decoration”: each component must be connected to an empirical effect.

## Current empirical story
The original global/fixed fusion did not beat TCN-CE overall. Diagnostics showed the consistency score was strongly correlated with classifier attack probability and global fusion worsened calibration. This is a useful negative finding.

A validation-trained selective combiner was then evaluated on a later untouched confirmation block. Mean results across seeds:

| Metric | TCN-CE | Selective combiner | Delta |
|---|---:|---:|---:|
| Accuracy | 0.991833 | 0.993583 | +0.001750 |
| Balanced accuracy | 0.979889 | 0.985238 | +0.005349 |
| Macro-F1 | 0.980177 | 0.985102 | +0.004925 |
| Macro-AP | 0.996898 | 0.989576 | -0.007322 |
| Macro-AUC | 0.999399 | 0.997733 | -0.001666 |

Thus the consistency mechanism improves hard classification decisions but currently sacrifices ranking quality. The manuscript should not claim universal superiority.

## Reviewer-safe claim
“CAN-ATCNet investigates whether predictive CAN-behavior inconsistency provides complementary evidence to supervised temporal attack classification. Across three seeds, selective integration of the residual-derived consistency signal improved confirmation-set accuracy, balanced accuracy and macro-F1 relative to TCN-CE, while AP/AUC results show that indiscriminate or uncalibrated integration can degrade ranking quality.”

## Claims NOT currently supported
- State-of-the-art performance.
- Superior performance on every metric.
- External generalization to unseen vehicles or independent captures.
- Zero-day/unknown-attack detection.
- First-ever dual-head or multitask CAN IDS.
- Raw residual score itself is the fused signal.
- Independent external validation.

## Remaining novelty-defense work
1. Preserve the current accuracy results rather than tuning repeatedly on confirmation data.
2. Produce clean ablation and mechanism figures.
3. Report calibration, per-class effects and paired error analysis.
4. Add computational cost/latency/parameter analysis.
5. Evaluate on an independent dataset or capture if the final paper claims generalization.
6. Conduct a systematic literature matrix specifically comparing predictive objectives, residual/anomaly heads, fusion rules, attack-type classification and evaluation protocols.

## Bottom line
The work has a defensible research contribution if framed as **predictive-consistency evidence for discriminative CAN intrusion classification, with selective integration and mechanism-level ablation**. The novelty is not the number of heads and not the near-99% accuracy.
