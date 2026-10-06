# Reproducibility and Artifact Audit

## Executive finding
The repository contains the source notebook and automation code, but it does **not currently contain the complete outputs produced on RunPod**.

At the time of this audit:
- `HCRL_Next_Experiment_3Seeds.ipynb` is present in the repository.
- GitHub Actions / RunPod automation is present.
- `results/` contains only `.gitkeep`.
- GitHub Actions run 37531969596 has zero uploaded artifacts.
- The repository notebook has some embedded outputs, including one image output, but it is not a complete archive of the latest RunPod experiment.
- Paper-critical CSV metrics from the confirmation run were printed into workflow logs and are therefore recoverable.
- RunPod-local checkpoints, NPZ prediction bundles and generated figures that were not printed or committed are no longer available after Pod deletion.

## What is safely preserved
1. Source notebook and implemented model/training/evaluation code.
2. Git history of methodology and automation changes.
3. Workflow logs for completed GitHub Actions runs.
4. Numeric CSV/JSON outputs explicitly printed in workflow logs.
5. Dataset source/bootstrap logic and experiment configuration encoded in the notebook/workflow.
6. Key confirmation metrics recovered into the repository after this audit.

## What is NOT proven to be preserved
1. Every generated PNG/PDF/SVG figure.
2. Every trained `.pt` checkpoint.
3. Every `.npz` prediction/component file.
4. The complete RunPod output directory.
5. Executed notebook with every latest output embedded.
6. A single immutable artifact bundle containing code + config + results + figures.

## Consequence
The numerical findings are recoverable and the experiment is reproducible from code, but the repository is not yet a complete paper artifact package. Any figure that depended on deleted RunPod-only files may need to be regenerated.

## Required archival standard for future runs
Every paper run should create a compact immutable bundle before Pod deletion:
- protocol/config JSON;
- exact commit SHA;
- environment/package manifest;
- dataset provenance/hash manifest;
- all summary CSV/JSON tables;
- per-class metrics and confusion matrices;
- calibration/error-analysis tables;
- paper figures (PNG + vector PDF/SVG where practical);
- prediction outputs needed to reproduce plots;
- executed notebook or a generated HTML report;
- logs;
- optional checkpoints, depending on size.

The GitHub workflow should upload this bundle using a GitHub Actions artifact before deleting the Pod. Paper-critical small files should also be committed under a versioned `results/` directory or attached to a release.

## Current paper-readiness status
- Code provenance: GOOD.
- Numeric-result provenance: RECOVERABLE / PARTIALLY ARCHIVED.
- Figure provenance: INCOMPLETE.
- Model-checkpoint provenance: INCOMPLETE.
- Reproducibility from source: GOOD, subject to public dataset availability and environment compatibility.
- Immutable experiment bundle: MISSING.

## Recommended next non-scientific task
Do not change or tune the model. Patch the workflow so the next reproducibility run archives the complete paper bundle. Then regenerate figures/tables from frozen code and frozen protocol without using those outputs for further model selection.
