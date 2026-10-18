# Paper 3 | Authoritative Penn Action reviewer-evidence protocol | 2026-10-19

**Status:** Current protocol. `paper3/PROTOCOL_DRAFT.md` is historical and describes the unrelated UTD-MHAD draft. Do not use it for this Penn Action experiment. Paper 2 code and experiments are frozen and remain separate.

## Research framing
Post-training, baseline-preserving adaptation of a pretrained RGB/annotated-pose Transformer, focusing on explicit complete modality unavailability. Training supervision of a reliability head and its incremental predictive value are separate questions.

## Data and leakage
- Penn Action: 2,326 clips, 15 labels, official train pool 1,258 / test 1,068.
- Split official train into 1,006/252 by seed 42, 123, 2026. Fixed class ordering.
- 16 RGB frames; 112 x 112 ImageNet normalization, frozen ResNet-18 512-dimensional output, 13 2D pose joints.
- The current method was already designed after reviewing Penn Action test numbers; this is a **retrospective exploratory extension**, not independently preregistered confirmation. Do not claim test-set independence of study design.
- Pose is dataset-provided annotation. Missing-RGB-with-pose means *observed input absent but oracle-quality pose provided*, NOT verified tolerance of physical camera outage.

## Single-run design
A preflight unit-test gate on GitHub-hosted CPU. One RunPod RTX4090 job for all actual training/evaluation and complete artifact return.

Same baseline per seed and V2 reliability router. Reuse the same trained V2 specialists for inference ablations so architecture/training differences do not confound the **principal** V2-versus-simple-switching comparison.

Compare:
- legacy_zero: frozen baseline with zero-filled absent inputs.
- specialist_rule: same trained experts, no quality mixing, clean fused reference if both observed.
- specialist_equal: same trained experts, equal mixing when both observed.
- v2: trained reliability-aware routing.
- specialist_quality_no_supervision: **inference-only gate replacement, NOT a retrained no-supervision model**. Do not call this an unsupervised training ablation.

Under explicit complete modality absence, V2 and specialist_rule are mathematically the same single-expert route. Any reported difference is a BUG. Learned reliability cannot be credited with complete-loss accuracy gain. Distinguish specialist routing and reliability performance.

## Failure/evaluation coverage
- Seen RGB: frame_dropout, blur, body_occlusion, temporal_gap, low_light.
- Held-out RGB operator types: compression and irregular_sampling. Excluded from training augmentation.
- RGB and pose corruption severities: 0.25, 0.50, 0.75. Clean and complete RGB/pose absence separately.
- Pose faults: gap, joint dropout, coordinate noise.
- Three trained seeds. For each condition write per-video labels/predictions/confidence and per-seed metrics.
- Metrics: accuracy, macro-F1, NLL, ECE; reliability AUROC/AUPRC **for known synthetic corruption-category targets** (not equivalent to predictive trustworthiness).
- Stratified paired bootstrap by class *within each seed*. Exploratory exact McNemar and Holm correction. Do not treat seed predictions on the same test clips as independent videos.
- Parameters (total and trainable) and measured feature-to-logit inference latency. Does **not** measure end-to-end video preprocessing, pose extraction, or deployment energy cost.
- Record train/validation epochs, source commit, dataset manifest checksum, checkpoint weights, and GPU deletion evidence.

## Known missing prerequisites
- A **second independent compatible RGB + pose dataset** and a valid mapping/annotation protocol are not verified. This job explicitly reports it as NOT PERFORMED; do not claim external generalization.
- Real predicted keypoints with confidence scores are not available for this verified Penn Action pipeline. Their evaluation is NOT PERFORMED; do not use ground truth as a proxy.
- Genuine trained "reliability head without supervision" would require an additional controlled training variant; the inference-only sensitivity variant is not that experiment.
- Incomplete/uneven physically meaningful severity definitions; all operators are synthetic controls.

## GPU controls
- Only a single one-shot Actions trigger, no automatic retries and no repeating pushes to the trigger path.
- GitHub Actions for orchestration and artifacts only, training on RunPod.
- CPU syntax/regression preflight before GPU rental, GPU remote smoke before complete run.
- Hard wall clock upper bound; durable encrypted archive transfer confirmed and validated before pod termination.
- Pod deletion attempted regardless of success or failure; failure loudly exposed. External runner failure may still require manual RunPod inspection.

**Acceptance:** six valid baseline/V2 checkpoints, per-video predictions for all tested conditions and seeds, uncertainty tables, reliability metrics, efficiency records, checksum/provenance, actual successful transfer, and confirmed pod deletion. No implicit scientific novelty or acceptance guarantees.
