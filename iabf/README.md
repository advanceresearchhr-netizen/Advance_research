# IABF Multi-PIE 50-image experiment: GitHub validation branch

**Status:** The original 50-image IABF experiment script is committed to `iabf/iabf_multipie_complete.py` on this branch. Local Python compilation and static 50-image guard checks passed on 2026-10-20. GitHub Actions execution remains unconfirmed; the smoke workflow is not a GPU or scientific validation.

The default branch is unchanged. The existing HCRL and other research projects are unaffected.

## Evaluation design
- Exactly 50 unique adaptation images overall, 10 identities × 5 samples each.
- Identity-disjoint adaptation, validation and test identities.
- Pretrained InceptionResNetV1 checkpoint and saved IABF checkpoint.
- Frontal enrollment, left/right rotated probes from Multi-PIE.
- Report per-camera Rank-1, ROC, EER, FAR-related measures, image quality robustness and ablations.
- Report raw coverage, checkpoint hashes, run seed and uncertainty.
- Do not commit Multi-PIE subject images or licensed data to GitHub.

## Run on GPU environment
```bash
python iabf/iabf_multipie_complete.py \
  --manifest /data/multipie_manifest.csv \
  --baseline /models/best_standard_recognizer.pth \
  --proposed /models/best_casia_recognizer.pth \
  --output /results/iabf_seed42 --seed 42 --epochs 30
```

The CI checks only syntax and guards, **not scientific or dataset-specific correctness**. A validated camera-to-yaw manifest and checkpoint compatibility audit are mandatory before publishing results.
