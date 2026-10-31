# Paper 3: Failure-Resilient Multimodal Human Action Recognition
Status: **methodology planning only; no experiments launched**.
## Central question
Can reliability-aware multimodal HAR maintain trustworthy recognition and calibrated uncertainty under realistic, controlled RGB and skeleton sensor degradation, compared with a frozen Paper 2 reference?
## Isolation
- Work only under `paper3/` on branch `paper3/failure-resilience`.
- Paper 2 is a frozen reference. Do not edit `paper2/`, its workflows, outputs, or published findings; do not change default branch `main`.
- Base reference: `ab2a0319220ce114ba22f340a9eaae33ed6915eb` (7 October 2026). Pin this SHA for all Paper 2 comparisons, rather than tracking moving `main`.
- Reuse of code, data preparation, and metrics must be disclosed; no duplicate scientific claims, figures, or results described as new.
## Planned scientific contribution (hypotheses, not results)
1. Model input-conditional reliability under both isolated and simultaneous sensor faults.
2. Distinguish true observation quality from learned attention weights; evaluate calibrated confidence and selective prediction.
3. Quantify clean/robustness trade-offs, out-of-distribution or unseen degradation generalization, latency, and interpretability preservation.
A novel architecture is **not** claimed until a dated 2024–2026 literature gap matrix is completed and reviewed.
## Immediate deliverables
- `REUSE_REGISTER.md`: provenance and allowed reuse.
- `PROTOCOL_DRAFT.md`: preregistration-style experimental plan.
- `RISKS_AND_GATES.md`: barriers to leakage, repeat publication, compute spend, and result loss.
## Execution status
No Paper 3 training, Actions dispatch, GPU rental, dataset downloads, or results are implied by this folder.
