# Paper 3: Failure-resilient Penn Action RGB–pose recognition

Reference: user-uploaded original Penn Action ResNet-18 + Transformer notebook, held in the private archive. The original Paper 2 notebook is NOT included in the public source tree. Never use the UTD-MHAD GitHub implementation as the scientific baseline.

## Source package
The primary code is in `paper3/src/paper3`. The model uses frozen ImageNet ResNet-18 visual features [T=16,512] and Penn Action annotated pose [T=16,13,2]; separate 2-layer temporal Transformers followed by learned fusion. Paper 3 adds quality-adaptive fusion and representation residual.

## Verify
```sh
cd paper3/src
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m paper3.build_manifest --data-root /path/to/Penn_Action --out penn_manifest.csv
python -m paper3.run --manifest penn_manifest.csv --data-root /path/to/Penn_Action --out output --seed 42 --model paper3 --device cuda --fault blur --severity .25
```
Use `--limit 2 --epochs 1` for a real-data smoke test and never report it as experimental performance. Full official test inference is explicitly gated by `--evaluate-test --allow-test`.

## Research limitations
- Original checkpoint and split CSV provenance is not yet independently recovered; exact historical training reproduction is not certified.
- Dataset archive must be downloaded and integrity-checked in the training environment.
- Existing feature-space training augmentation is not full image-level robust training. Visual faults are applied to original frames BEFORE ResNet extraction during evaluation.
- The new quality-aware mechanism is a *hypothesis*, not a verified novelty or superiority claim.
- Current tests use synthetic inputs; full real Penn Action GPU smoke and performance evaluation are pending.
- RunPod launch is blocked until durable artifact upload and unconditional pod cleanup are operational and tested.
- No academic citation or numerical comparison should use unpublished Paper 2 as published external work.
