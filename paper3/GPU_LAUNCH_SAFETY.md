# Paper 3 RunPod launch safety contract

**Do not run the original UTD-MHAD workflow.** This research project uses Penn Action and the uploaded RGB+pose ResNet-18/Transformer baseline.

## Preconditions (must all be true before paid Pod creation)
- A complete, reviewed Penn Action Paper 3 source tree and unit tests are present at the pinned commit.
- Dataset download and integrity validation are verified for the official Penn Action archive; cache or resumable download is recommended.
- A genuine Penn Action smoke test succeeds before multi-seed training.
- Exact original train/validation split logic and separate official test protocol are checked against the user's Colab notebook.
- Results/checkpoints have a tested external export destination **before** any pod is launched. Logs alone do not preserve learned model weights.
- A single RunPod pod is created; no automatic retries or overlapping launches.
- A bounded timeout is configured in BOTH the pod training process and GitHub Actions poller.
- An unconditional cleanup step terminates the pod on success, failure, and timeout; deletion failure must be visible, never silently treated as success.
- GitHub Actions is used only for orchestration and concise logs, never long-running training or large data processing.
- Secrets are passed through GitHub Actions secrets, never printed, saved as artifacts, or committed.

## Current status
**BLOCKED: do not launch.** The source-only Paper 3 implementation, published dataset tests, and durable checkpoint export are not yet complete in this branch. The existing repository has a UTD-MHAD workflow, which is scientifically unrelated to the uploaded Paper 2 baseline.

## Scope of scientific evaluation
- Train original Penn Action ResNet-18/Transformer model and Paper 3 separately with matched seeds and identical splits; distinguish rerun baseline from historical Paper 2.
- Evaluate frame loss, temporal gaps, pixel blur, body occlusion, low-quality frames, irregular sampling, and annotated 2D pose failure under matched severity protocols.
- Ensure visual corruptions are applied at image level **before** frozen ResNet-18 extraction.
- Record accuracy, macro-F1, calibration, per-corruption degradation, ablations, uncertainty intervals and negative results. Never claim originality or numerical superiority without fair literature comparison.
