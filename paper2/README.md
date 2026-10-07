# Paper 2 execution

Gate 1 novelty freeze is in `GATE1_NOVELTY_FREEZE.md`.

## Data gate
Use the official NTU RGB+D 60 research dataset. Do not commit or redistribute NTU data in this repository.

Expected inputs:
- RGB video directory containing official sample filenames such as S001C002P003R002A013_rgb.avi
- skeleton directory containing corresponding .skeleton files

Run the zero-cost preflight before any RunPod launch:

```
python paper2/preflight_ntu60.py --rgb /data/ntu60/rgb --skeleton /data/ntu60/skeleton
```

A paid experiment must not start unless the preflight passes, the CS/CV split manifest is generated, a smoke test passes, and result/artifact archival is configured.

## Scientific rule
The central claim is multi-level explanation-to-decision faithfulness across synchronized RGB, skeleton, temporal and modality evidence. Classification accuracy is supporting evidence, not the novelty claim.
