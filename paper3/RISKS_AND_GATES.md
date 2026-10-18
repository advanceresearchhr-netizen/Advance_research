# Paper 3 research gates and risks
## Stop conditions
- No edits to Paper 2 or the `main` reference branch.
- No Actions dispatch, RunPod provisioning, paid compute, or model training under this setup change.
- No final-test tuning or use of held-out test samples for model selection.
- No Q1, SCIE or novelty guarantee without literature verification and experiments.
## Known inherited operational risks
At the frozen Paper 2 snapshot, RunPod LOSO workflow uploads only a log while results remain on the pod and the pod is then deleted; it does not enforce prerequisite workflow success before pod creation. Do not clone this behavior. A Paper 3 workflow must fail closed on missing credentials/prerequisites, pin the exact git SHA, archive weights/predictions/metrics/environment/logs durably, verify integrity, and clean up billing resources even on failure.
## Open checks
- Current status of repository Actions permissions, branch rules, required secrets, and last completed jobs.
- Existence/provenance of Paper 2 model checkpoints and artifact retention.
- Literature gap uniqueness in uncertainty-aware multimodal HAR and sensor-degradation robustness (2024–2026).
- Corruption physical validity, dataset licensing, compute budget and external evaluation feasibility.
- Publication overlap and appropriate reference/attribution to completed Paper 2.
## Milestones
Gate A: verified novelty matrix and distinct research claim.
Gate B: code/data provenance review and test-only safeguards.
Gate C: CPU smoke and integrity checks, no paid compute.
Gate D: explicit go/no-go authorization for GPU and archival testing.
Gate E: complete experiments and independent manuscript evaluation.
