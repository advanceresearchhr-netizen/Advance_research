# IABF RunPod GPU workflow (isolated)

**Workflow:** `.github/workflows/iabf-runpod.yml` on the `research/iabf-multipie-50image-ci` branch.

This workflow is **manual-dispatch only**. It does not change, call, cancel, or modify any HCRL / Paper-2 workflow. It uses a separate job, concurrency group, pod name, working directory, and artifact prefix.

## Required repository secrets

- `RUNPOD_API_KEY`: existing RunPod API key (reuses the HCRL infrastructure).
- `KAGGLE_USERNAME` and `KAGGLE_KEY`: Kaggle API credentials for dataset access.
- `IABF_ASSETS_DATASET`: exact Kaggle private dataset slug `owner/slug` with the original **best_standard_recognizer.pth**, **best_casia_recognizer.pth**, and **multipie_manifest.csv**.

The Kaggle dataset `aliates/multi-pie` is downloaded inside the GPU pod. The **private assets Kaggle dataset must be restricted** because it contains trained weights. Never place raw pretrained checkpoints or Kaggle credentials into public GitHub commits.

### REQUIRED scientific input: multipie_manifest.csv

Must contain `path,subject,camera,yaw_deg`; may additionally contain `session,illumination`. Paths must be relative to the extracted public dataset root. Multi-PIE camera IDs and signed yaw degrees must be verified against real camera metadata, not inferred from arbitrary filename digits. One person across sessions must share the same subject ID. Camera-to-yaw mapping must be one-to-one and signed consistently.

**The workflow cannot generate a scientifically valid manifest from an unknown Kaggle folder structure.** It intentionally fails when that input is absent or invalid. It does not invent camera angles.

### Before dispatching

1. Add the three files to a private Kaggle dataset and save its owner/slug as `IABF_ASSETS_DATASET`.
2. Add Kaggle API secrets to **Settings → Secrets and variables → Actions**.
3. Confirm the repo's existing `RUNPOD_API_KEY` has budget and GPU provisioning rights.
4. Open **Actions → IABF Multi-PIE 50-image RunPod GPU → Run workflow**; select branch `research/iabf-multipie-50image-ci`, GPU type, seed, epochs.

## Execution

1. Free GitHub-hosted preflight validates secrets, experiment source and Kaggle asset listings.
2. Only after preflight, start a dedicated RunPod pod.
3. Download datasets, validate the manifest and ensure CUDA works.
4. Adapt the IABF gate on exactly **50** images; evaluate identity-disjoint frontal-to-rotated recognition and ablations.
5. Transfer compact results (CSV/JSON/PNG) as a checksummed compressed log bundle, then upload `iabf_results.zip` and `iabf.log` as GitHub Actions artifacts.
6. Delete **only** the pod whose ID was created by this workflow.

**Limitations:** GitHub's preflight checks file existence in Kaggle, not the scientific mapping of camera angles. First GPU execution still needs a close audit, and quality of the resulting research claims is not automatic. For complete checkpoint/feature caches, use persistent storage; they are deliberately excluded from the size-capped GitHub artifact bundle. RunPod time and Kaggle downloads may exceed workflow timeout for large data mirrors.

## Security / billing

Credential fields are never printed. GitHub Actions secrets must be set before manual dispatch. Do not set `cancel-in-progress: true` for paid GPU experiments, as cancellation during pod creation can prevent cleanup. A paid RunPod pod can still outlive a force-cancelled or failed GitHub runner; check your RunPod dashboard if any run stops unexpectedly.
