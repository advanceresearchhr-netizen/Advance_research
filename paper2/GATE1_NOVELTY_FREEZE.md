# Paper 2 Gate 1: Novelty and Defensibility Freeze

## Candidate paper
Faithful Explainable Multimodal Human Action Recognition

## Decision
PROVISIONALLY DEFENSIBLE, but only under a narrow claim. Do not claim novelty for RGB+skeleton fusion, VideoMAE, ST-GCN, joint/bone/motion streams, adaptive modality weighting, temporal Transformers, key-frame selection, Grad-CAM, or perturbation faithfulness by themselves.

## Narrow central claim
A synchronized multimodal HAR framework in which explanations are produced at four evidence levels (temporal segments, RGB regions, skeleton joints, and modality contribution) and are quantitatively tested against the same classifier decision using modality-aware interventions.

The research question is:
"Are the temporal, spatial, skeletal, and modality-level explanations of a multimodal HAR classifier faithful to the evidence that actually changes its prediction?"

## Literature pressure
Existing work already covers:
- quantitative faithfulness/stability evaluation for skeleton HAR on NTU RGB+D;
- CAM/Grad-CAM/SHAP joint-level explanation for skeleton HAR;
- perturbation-based faithfulness for skeleton HAR;
- RGB+skeleton multimodal recognition, dynamic/score-weighted fusion, attention and Grad-CAM visualization.

Therefore the contribution must be cross-modal and decision-consistency based, not "XAI for HAR" in general.

## Proposed contribution
1. One multimodal classifier with synchronized RGB and skeleton evidence.
2. Four aligned explanation levels: time, RGB region, joint, modality.
3. Intervention-based Explanation-Decision Consistency evaluation:
   - remove/perturb top-ranked evidence;
   - matched random evidence baseline;
   - bottom-ranked evidence baseline;
   - measure class-confidence drop and label stability;
   - insertion curves where valid.
4. Plausibility-aware perturbations:
   - skeleton joints are perturbed rather than arbitrarily zeroed when possible;
   - RGB evidence uses masking/blur/inpainting-style intervention with explicit caveat;
   - modality deletion uses missing-modality handling learned/validated without test tuning.
5. Report faithfulness separately by modality and explanation level before any aggregate score.

## Important restraint
Do not invent a new "faithfulness score" until existing metrics are fully compared. Primary results should use established deletion/insertion/PGI/PGU-style measures and random controls. A composite score is secondary only if mathematically distinct and justified.

## Frozen evaluation protocol
Dataset: NTU RGB+D 60, official Cross-Subject and Cross-View splits.
Primary multimodal inputs: synchronized RGB + provided 3D skeleton. Motion is derived evidence, not a separately claimed sensor modality.
Recognition metrics: top-1 accuracy, macro-F1, per-class recall, confusion matrix.
XAI metrics: deletion AUC / confidence drop, insertion AUC where meaningful, top-vs-random perturbation effect, bottom-vs-top effect, explanation stability under small plausible perturbations.
Statistical reporting: paired bootstrap confidence intervals across samples; effect sizes; per-class analysis.
Baselines:
- RGB-only
- skeleton-only
- ordinary RGB+skeleton fusion
- full multimodal classifier
XAI controls:
- random attribution
- Grad-CAM / Integrated Gradients where compatible
- skeleton attribution baseline
Ablations:
- without motion-conditioned interaction
- without temporal evidence selection
- without explanation-decision consistency evaluation/training component if a training loss is ultimately introduced
- modality-specific interventions

## Leakage / integrity rules
- Official NTU split IDs only.
- No test-driven architecture or hyperparameter selection.
- Fit normalization and any learned missing-modality/fusion calibration on training/validation only.
- Freeze all thresholds before final test evaluation.
- Keep development and final confirmatory evaluation separate.
- Preserve seeds, environment, exact commit SHA, configs, metrics, predictions, figures, and logs.

## Gate 2 entry criteria
Before paid GPU:
1. Official NTU RGB+D RGB and skeleton data are available under the approved research license.
2. Preflight validates synchronized sample IDs and official CS/CV split construction.
3. A small CPU/GPU smoke test passes.
4. Artifact upload/archival is configured before Pod launch.

## Gate 1 verdict
Proceed. The paper is not revolutionary, and it does not need to be. The defensible novelty is the synchronized, multi-level, quantitatively verified explanation-to-decision relationship across RGB, skeleton, temporal, and modality evidence. This remains a candidate claim, not a "first-ever" claim, until the final literature matrix is complete.
