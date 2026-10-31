# Paper 3 reuse and provenance register
**Frozen Paper 2 source:** `advanceresearchhr-netizen/Advance_research`, commit `ab2a0319220ce114ba22f340a9eaae33ed6915eb`, directory `paper2/`.
Reuse classifications below are **planned**, not evidence of copied or executed assets. Verify availability and licenses before use.

| Source component | Planned Paper 3 treatment | Provenance / safeguard |
|---|---|---|
| `paper2/protocol.yaml` | Reference experimental protocol | Cite Paper 2; preserve subject split where directly comparable |
| `paper2/experiment.py` | Candidate encoders, loader, recognition baselines | Pin source SHA; inspect implementation and licenses before adaptation |
| `paper2/dev_loso.py` | Development-only subject-wise LOSO procedure | Maintain subject isolation, no final-test-driven tuning |
| `paper2/validate_utd_archives.py` | Dataset pairing/data integrity checks | Do not edit Paper 2 file; import or duplicate with attribution if necessary |
| `paper2/real_data_smoke.py` | Reference input/gradient smoke behavior | New Paper 3 tests reside in `paper3/` |
| `paper2/gpu_ready.py` | Reference validation and output contract | Extend new robustness/calibration checks separately |
| `paper2/GPU_RUN.md` | Reproducibility, archiving and split requirements | Do not inherit existing RunPod archival weaknesses |
| `.github/workflows/paper2-*.yml` | Read-only infrastructure reference | Never edit or dispatch as part of Paper 3 startup |
| UTD-MHAD RGB and skeleton data | Shared raw dataset, subject-disjoint analysis | Official data source; verify use conditions and checksum |
| Paper 2 trained checkpoints | Optional frozen baseline **if extant and verified** | Confirm exact SHA, training split and seed; never assume weights exist |
| Paper 2 published tables/figures | Context/citation only | Do not represent reused outputs as new experiments |
| Paper 2 explanation code | Optional comparative baseline | Distinct failure-resilience question; document code provenance |
## New work required for Paper 3
- Defined fault model and severity calibration, including real or plausibly simulated degradations.
- Sensor reliability and uncertainty estimation with independent calibration validation.
- Recovery/fusion strategy distinct from Paper 2, with negative controls.
- Paired clean/corrupted analysis, unseen degradation testing, calibration, abstention/coverage and compute accounting.
- Independent ablation and scientific novelty assessment against recent related work.
## Publication integrity
Disclose overlap and prior work in the manuscript and cover letter as required by target journal. Distinct hypotheses, analyses and conclusions are mandatory. Results must not be fabricated or duplicated.
