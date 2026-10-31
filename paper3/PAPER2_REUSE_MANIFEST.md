# PAPER2_REUSE_MANIFEST.md
Status: PROVISIONAL SOURCE AUDIT; NOT AN IMMUTABLE PAPER 2 SNAPSHOT. Do not commence training.
Authoritative evidence: user-uploaded ChatGPT Library `Paper2_Corrected_Standalone_ThreeSeed.py` (file_000000000d048210a5b8b08824bc15a0), `Paper2_MTO_Candidate_Validation.py` (file_00000000ebac821089fd939282da7f24), and `Paper2_Full_Journal_Manuscript_Draft.docx` (file_0000000048648210a013673d225caca4). Preserve originals unchanged. Library IDs are catalog identifiers, **not file content hashes**.
## Scientific reference
- Penn Action, 15 actions, 2326 videos; official training 1258, official test 1068; per seed training 1006, validation 252. Per-seed partitions and official test should be verified from original CSVs when recovered.
- Image input: 16 RGB frames, 112x112, ImageNet mean/std, frozen ResNet-18 pretrained embedding of 512 features/frame.
- Pose: 13 annotated 2D joints per frame, coordinate normalization by original image dimensions to [-1,1], producing 16x13x2.
- Separate temporal Transformer branches: input dimensions 512/26; model dimension 128, 4 heads, 2 encoder layers, feedforward 256, GELU, dropout 0.1, LayerNorm, learned positional embeddings.
- Fusion: feature-wise 128-dimensional sigmoid gate on concatenated representations, g*r + (1-g)*p. Classifier: linear 128->128, GELU, dropout 0.25, linear 128->15.
- Seeds 42,123,2026; validation scripts demand per-seed checkpoint-stored accuracy and macro-F1 match recomputation to <1e-6.
- Paper 2 research claim: intervention-conditioned attribution faithfulness of pose joints, NOT failure-resilient recognition. Manuscript reports mean official test 91.07% accuracy and 90.53% macro-F1; these are manuscript-reported figures requiring raw checkpoint/results verification, not independently rerun here.
## Required original file paths (not yet found in accessible Library/Drive search)
- /content/Paper2/PennAction/Penn_Action/frames and labels
- /content/Paper2/splits/train_seed{42,123,2026}.csv and val_seed{42,123,2026}.csv
- /content/Paper2/temporal_runs/temporal_fusion_seed42.pt, ordered_seed123.pt, ordered_seed2026.pt
- Original training notebook/scripts, optimizer, learning-rate schedule, epoch count, checkpoint selection, environment/dependency versions, official-test manifest, raw run logs.
Do not claim immutable Paper 2 reference frozen until original files are located, archived read-only with SHA-256 digests, and versions/provenance recorded.
## Authorized reuse
- Reuse PennDataset normalization, sample alignment, 16-frame sampling, ResNet-18 frozen feature extraction, temporal Transformer, fusion/classifier, per-seed splits and metrics after provenance confirmation.
- New Paper 3 code must reside under isolated `paper3/` and be distinct from the original training implementation.
- Novel extensions: visual frame dropout, temporal gaps, blur, body-region occlusion, compression/low-quality video, irregular sampling; annotated keypoint corruption/dropout is legitimate as pose inputs are verified.
- Distinguish annotation-level pose corruption from deployable pose estimator errors; validate realism before experimental claims.
- Primary endpoints must address failure resilience and prediction reliability rather than repeat Paper 2 explanation benchmarks.
## Explicit exclusions
- GitHub `paper2/` UTD-MHAD `FaithfulHAR`, its subject splits and 3D skeleton model are NOT authoritative for this work.
- CAN-ATCNet and unrelated intrusion-detection/other HAR results are not valid baselines.
- Previous Paper 3 documents in this branch referring to UTD-MHAD are **superseded and must not be used**; their correction is a mandatory next action before implementation.
## Project gates
1. Recover original files and hash their exact bytes.
2. Create read-only, versioned scientific snapshot and verify original validation/test calculations.
3. Freeze Paper 3 corruption and training protocol with literature check.
4. Implement separate tests and obtain verified real-data smoke pass.
5. Obtain explicit authorization before paid GPU/RunPod provisioning.
